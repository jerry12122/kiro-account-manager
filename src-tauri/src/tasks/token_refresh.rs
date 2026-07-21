// Token 自动刷新后台任务
// 参考 Kiro IDE 源码实现；含 IDE 回写、invalid_grant 救援、冷却与三次停用

use crate::commands::common::{
    apply_refreshed_account_tokens, is_auth_error_message, is_token_expired,
    refresh_token_by_provider, token_needs_refresh, RefreshResult, REFRESH_LOOP_INTERVAL_SECONDS,
};
use crate::core::account::Account;
use crate::kiro::token_sync::{
    apply_ide_tokens_to_account, record_refresh_auth_failure, should_skip_refresh_cooldown,
    sync_kam_tokens_to_ide_if_matches, REFRESH_AUTH_FAILURE_STRIKES_TO_DISABLE,
};
use crate::state::AppState;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager};
use tokio::time::{interval, Duration};

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn account_label(account: &Account) -> String {
    account
        .email
        .as_deref()
        .or(account.user_id.as_deref())
        .unwrap_or("Unknown")
        .to_string()
}

/// Token 刷新服务
pub struct TokenRefreshService {
    app_handle: AppHandle,
    /// 连续认证失败计数（内存；成功清零）
    auth_failure_strikes: Mutex<HashMap<String, u32>>,
    /// 上次成功刷新的 Unix 秒（冷却）
    last_success_unix: Mutex<HashMap<String, u64>>,
    /// 正在刷新的账号（互斥）
    in_flight: Mutex<HashSet<String>>,
}

impl TokenRefreshService {
    /// 创建新的 Token 刷新服务
    pub fn new(app_handle: AppHandle) -> Self {
        Self {
            app_handle,
            auth_failure_strikes: Mutex::new(HashMap::new()),
            last_success_unix: Mutex::new(HashMap::new()),
            in_flight: Mutex::new(HashSet::new()),
        }
    }

    fn try_acquire(&self, account_id: &str) -> bool {
        let mut guard = match self.in_flight.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if guard.contains(account_id) {
            return false;
        }
        guard.insert(account_id.to_string());
        true
    }

    fn release(&self, account_id: &str) {
        let mut guard = match self.in_flight.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.remove(account_id);
    }

    fn clear_strikes(&self, account_id: &str) {
        let mut guard = match self.auth_failure_strikes.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.remove(account_id);
    }

    fn bump_strikes(&self, account_id: &str) -> (u32, bool) {
        let mut guard = match self.auth_failure_strikes.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let current = guard.get(account_id).copied().unwrap_or(0);
        let (next, disable) = record_refresh_auth_failure(current);
        guard.insert(account_id.to_string(), next);
        (next, disable)
    }

    fn mark_success_cooldown(&self, account_id: &str) {
        let mut guard = match self.last_success_unix.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.insert(account_id.to_string(), now_unix());
    }

    fn in_cooldown(&self, account_id: &str) -> bool {
        let guard = match self.last_success_unix.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        should_skip_refresh_cooldown(guard.get(account_id).copied(), now_unix())
    }

    /// 启动后台刷新循环
    pub fn start(self) {
        tauri::async_runtime::spawn(async move {
            let mut interval_timer = interval(Duration::from_secs(REFRESH_LOOP_INTERVAL_SECONDS));

            loop {
                interval_timer.tick().await;

                if let Err(e) = self.refresh_expiring_tokens().await {
                    log::error!("Token refresh loop error: {}", e);
                }
            }
        });
    }

    /// 检查并刷新即将过期的 token
    async fn refresh_expiring_tokens(&self) -> Result<(), String> {
        let accounts = {
            let state = self.app_handle.state::<AppState>();
            let mut store = state
                .store
                .lock()
                .map_err(|_| "Failed to acquire account store lock".to_string())?;
            store.reload();
            store.accounts.clone()
        };

        let mut any_changed = false;

        for account in accounts {
            if account.status == "invalid" || account.status == "banned" {
                log::debug!(
                    "[TokenRefresh] skip reason=status_invalid account={}",
                    account_label(&account)
                );
                continue;
            }

            let Some(ref expires_at) = account.expires_at else {
                continue;
            };

            if !token_needs_refresh(expires_at) {
                log::debug!(
                    "[TokenRefresh] skip reason=not_expiring account={} expires_at={}",
                    account_label(&account),
                    expires_at
                );
                continue;
            }

            if self.in_cooldown(&account.id) {
                log::debug!(
                    "[TokenRefresh] skip reason=cooldown account={} expires_at={}",
                    account_label(&account),
                    expires_at
                );
                continue;
            }

            if !self.try_acquire(&account.id) {
                log::debug!(
                    "[TokenRefresh] skip reason=in_flight account={}",
                    account_label(&account)
                );
                continue;
            }

            let changed = self.refresh_one_account(&account, expires_at).await;
            self.release(&account.id);
            if changed {
                any_changed = true;
            }
        }

        if any_changed {
            let _ = self.app_handle.emit("accounts-updated", ());
        }

        Ok(())
    }

    /// 刷新单个账号；返回是否写库变更
    async fn refresh_one_account(&self, account: &Account, expires_at: &str) -> bool {
        let email_display = account_label(account);
        let provider = account.provider.as_deref().unwrap_or("Unknown");

        log::info!(
            "[TokenRefresh] attempt account={} provider={} expires_at={}",
            email_display,
            provider,
            expires_at
        );

        match refresh_token_by_provider(account).await {
            Ok(refresh_result) => {
                self.apply_success(account, &refresh_result, &email_display, false)
                    .await
            }
            Err(e) => {
                if e.starts_with("UPSTREAM_BLOCKED:") {
                    log::debug!(
                        "[TokenRefresh] upstream blocked refresh for {}",
                        email_display
                    );
                    return false;
                }

                log::error!(
                    "[TokenRefresh] refresh failed for {}: {}",
                    email_display,
                    e
                );

                if is_auth_error_message(&e) {
                    return self
                        .handle_auth_failure(account, expires_at, &email_display, &e)
                        .await;
                }
                false
            }
        }
    }

    async fn apply_success(
        &self,
        account: &Account,
        refresh_result: &RefreshResult,
        email_display: &str,
        from_rescue: bool,
    ) -> bool {
        let (snapshot, new_expires) = {
            let state = self.app_handle.state::<AppState>();
            let mut store = match state.store.lock() {
                Ok(s) => s,
                Err(poisoned) => poisoned.into_inner(),
            };
            store.reload();

            let Some(acc) = store.accounts.iter_mut().find(|a| a.id == account.id) else {
                return false;
            };

            // 救援路径允许用更新后的 RT；普通路径仍跳过陈旧结果
            if !from_rescue && acc.refresh_token.as_deref() != account.refresh_token.as_deref() {
                log::info!(
                    "[TokenRefresh] skipped stale refresh result for {}",
                    email_display
                );
                return false;
            }

            apply_refreshed_account_tokens(acc, refresh_result);
            let new_expires = acc.expires_at.clone().unwrap_or_default();
            let snapshot = acc.clone();

            if let Err(e) = store.try_save_to_file() {
                log::error!("Failed to save account after refresh: {}", e);
                return false;
            }
            (snapshot, new_expires)
        };

        self.clear_strikes(&account.id);
        self.mark_success_cooldown(&account.id);

        log::info!(
            "[TokenRefresh] success account={} expires_in={} new_expires_at={} from_rescue={}",
            email_display,
            refresh_result.expires_in,
            new_expires,
            from_rescue
        );

        match sync_kam_tokens_to_ide_if_matches(account, &snapshot).await {
            Ok(wrote) => {
                log::info!(
                    "[TokenRefresh] ide_sync account={} is_ide_current_and_wrote={}",
                    email_display,
                    wrote
                );
            }
            Err(e) => {
                log::warn!(
                    "[TokenRefresh] ide_sync failed for {} (tokens saved in KAM): {e}",
                    email_display
                );
            }
        }

        true
    }

    async fn handle_auth_failure(
        &self,
        account: &Account,
        expires_at: &str,
        email_display: &str,
        error: &str,
    ) -> bool {
        // 1) RT 已被他处更新 → 跳过
        {
            let state = self.app_handle.state::<AppState>();
            let mut store = match state.store.lock() {
                Ok(s) => s,
                Err(poisoned) => poisoned.into_inner(),
            };
            store.reload();
            if let Some(acc) = store.accounts.iter().find(|a| a.id == account.id) {
                if acc.refresh_token.as_deref() != account.refresh_token.as_deref() {
                    log::info!(
                        "[TokenRefresh] rescue skip: RT already updated elsewhere for {}",
                        email_display
                    );
                    self.clear_strikes(&account.id);
                    return false;
                }
            }
        }

        // 2) 若为 IDE 当前号（RT/AT/hash 匹配），从 IDE 救回凭证并重试一次
        let mut rescued = false;
        if let Some(local) = crate::kiro::ide::get_kiro_local_token().await {
            let is_ide_current =
                crate::kiro::token_sync::account_matches_ide_token(account, &local)
                    || (account.client_id_hash.as_ref().is_some_and(|h| !h.trim().is_empty())
                        && local.client_id_hash == account.client_id_hash);

            if is_ide_current {
                let retry_account = {
                    let state = self.app_handle.state::<AppState>();
                    let mut store = match state.store.lock() {
                        Ok(s) => s,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    store.reload();
                    if let Some(acc) = store.accounts.iter_mut().find(|a| a.id == account.id) {
                        if apply_ide_tokens_to_account(acc, &local) {
                            let _ = store.try_save_to_file();
                            rescued = true;
                            log::warn!(
                                "[TokenRefresh] rescue: imported IDE tokens for {} after {}",
                                email_display,
                                error
                            );
                        }
                    }
                    store
                        .accounts
                        .iter()
                        .find(|a| a.id == account.id)
                        .cloned()
                };

                if rescued {
                    if let Some(retry_account) = retry_account {
                        match refresh_token_by_provider(&retry_account).await {
                            Ok(refresh_result) => {
                                log::info!(
                                    "[TokenRefresh] rescue retry succeeded for {}",
                                    email_display
                                );
                                return self
                                    .apply_success(
                                        &retry_account,
                                        &refresh_result,
                                        email_display,
                                        true,
                                    )
                                    .await;
                            }
                            Err(retry_err) => {
                                log::warn!(
                                    "[TokenRefresh] rescue retry failed for {}: {}",
                                    email_display,
                                    retry_err
                                );
                            }
                        }
                    }
                }
            } else {
                log::debug!(
                    "[TokenRefresh] rescue skipped: account is not IDE current login ({})",
                    email_display
                );
            }
        }

        // 3) 累计 strike；满 3 且 token 已过期窗口才停用
        if !is_token_expired(expires_at) {
            log::warn!(
                "[TokenRefresh] auth failure but token not in expiry window yet; not disabling ({})",
                email_display
            );
            return false;
        }

        let (strike, should_disable) = self.bump_strikes(&account.id);
        log::warn!(
            "[TokenRefresh] auth failure strike={}/{} account={} rescued_attempted={}",
            strike,
            REFRESH_AUTH_FAILURE_STRIKES_TO_DISABLE,
            email_display,
            rescued
        );

        if !should_disable {
            return false;
        }

        let state = self.app_handle.state::<AppState>();
        let mut store = match state.store.lock() {
            Ok(s) => s,
            Err(poisoned) => poisoned.into_inner(),
        };
        store.reload();
        if let Some(acc) = store.accounts.iter_mut().find(|a| a.id == account.id) {
            // 他处已换了更新的 RT（非本次救援写入）→ 不误杀
            if !rescued && acc.refresh_token.as_deref() != account.refresh_token.as_deref() {
                log::info!(
                    "[TokenRefresh] skipped stale auth failure disable for {}",
                    email_display
                );
                self.clear_strikes(&account.id);
                return false;
            }

            acc.status = "invalid".to_string();
            acc.enabled = false;
            let _ = store.try_save_to_file();
            log::warn!(
                "[TokenRefresh] marked account {} as invalid after {} strikes",
                email_display,
                strike
            );
            return true;
        }

        false
    }
}

/// 启动 Token 刷新循环（供 main.rs 调用）
pub fn start_token_refresh_loop(app_handle: AppHandle) {
    log::info!("Starting token refresh background task");
    let service = TokenRefreshService::new(app_handle);
    service.start();
}
