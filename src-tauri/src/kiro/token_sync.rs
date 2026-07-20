//! KAM ↔ IDE token 同步与匹配（供 AutoSwitch / Token refresh loop 共用）

use crate::core::account::Account;
use crate::kiro::ide::{self, KiroLocalToken};

/// 把 KAM 内部 expiresAt（`%Y/%m/%d %H:%M:%S` 本地时区）转成 IDE 用的 RFC3339 UTC
pub fn kam_expires_at_to_ide_rfc3339(expires_at: Option<&str>) -> String {
    use chrono::TimeZone;
    if let Some(raw) = expires_at.map(str::trim).filter(|s| !s.is_empty()) {
        if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(raw, "%Y/%m/%d %H:%M:%S") {
            if let Some(local) = chrono::Local.from_local_datetime(&naive).single() {
                return local
                    .with_timezone(&chrono::Utc)
                    .to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
            }
        }
    }
    (chrono::Utc::now() + chrono::Duration::hours(1))
        .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

/// 用 KAM 账号上的 access/refresh 覆盖 IDE 本地凭证（同账号续命，不改 authMethod/provider）
pub async fn sync_kam_tokens_to_ide(account: &Account) -> Result<(), String> {
    let access = account
        .access_token
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "KAM 账号缺少 accessToken，无法回写 IDE".to_string())?;
    let refresh = account
        .refresh_token
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "KAM 账号缺少 refreshToken，无法回写 IDE".to_string())?;

    ide::update_kiro_local_tokens(
        access.to_string(),
        refresh.to_string(),
        Some(kam_expires_at_to_ide_rfc3339(account.expires_at.as_deref())),
    )
    .await
}

/// 本地匹配：账号是否为 IDE 当前登录号（不含网络 usage 反查）
///
/// 顺序：refreshToken 全等 → accessToken 全等 → accessToken 前缀(20) → clientIdHash
pub fn account_matches_ide_token(account: &Account, local: &KiroLocalToken) -> bool {
    if let Some(rt) = local.refresh_token.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        if account
            .refresh_token
            .as_deref()
            .map(str::trim)
            .is_some_and(|acc_rt| acc_rt == rt)
        {
            return true;
        }
    }

    if let Some(at) = local.access_token.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        if account
            .access_token
            .as_deref()
            .map(str::trim)
            .is_some_and(|acc_at| acc_at == at)
        {
            return true;
        }
        let prefix = &at[..at.len().min(20)];
        if !prefix.is_empty()
            && account
                .access_token
                .as_deref()
                .map(str::trim)
                .is_some_and(|acc_at| acc_at.starts_with(prefix))
        {
            return true;
        }
    }

    if let Some(hash) = local
        .client_id_hash
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        if account
            .client_id_hash
            .as_deref()
            .map(str::trim)
            .is_some_and(|h| h == hash)
        {
            return true;
        }
    }

    false
}

/// 若账号匹配 IDE 当前登录，则把 KAM token 回写 IDE；返回是否已尝试回写
pub async fn sync_kam_tokens_to_ide_if_current(account: &Account) -> Result<bool, String> {
    let Some(local) = ide::get_kiro_local_token().await else {
        log::debug!(
            "[TokenSync] skip IDE write: no local token ({})",
            account.email.as_deref().unwrap_or("未知")
        );
        return Ok(false);
    };

    if !account_matches_ide_token(account, &local) {
        log::debug!(
            "[TokenSync] skip IDE write: account is not IDE current login ({})",
            account.email.as_deref().unwrap_or("未知")
        );
        return Ok(false);
    }

    sync_kam_tokens_to_ide(account).await?;
    log::info!(
        "[TokenSync] wrote KAM access/refresh to IDE for {}",
        account.email.as_deref().unwrap_or("未知")
    );
    Ok(true)
}

/// 把 IDE 本地 RT/AT 救回 KAM 账号字段（不落盘；调用方负责 save）
pub fn apply_ide_tokens_to_account(account: &mut Account, local: &KiroLocalToken) -> bool {
    let mut changed = false;
    if let Some(rt) = local
        .refresh_token
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        if account.refresh_token.as_deref() != Some(rt) {
            account.refresh_token = Some(rt.to_string());
            changed = true;
        }
    }
    if let Some(at) = local
        .access_token
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        if account.access_token.as_deref() != Some(at) {
            account.access_token = Some(at.to_string());
            changed = true;
        }
    }
    if let Some(exp) = local.expires_at.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        if let Some(normalized) = crate::commands::common::normalize_expires_at(exp) {
            if account.expires_at.as_deref() != Some(normalized.as_str()) {
                account.expires_at = Some(normalized);
                changed = true;
            }
        }
    }
    changed
}

/// Token refresh 成功后的最短冷却（秒）——挡住每分钟轮换风暴
pub const REFRESH_SUCCESS_COOLDOWN_SECS: u64 = 30 * 60;

/// 连续认证失败多少次才停用
pub const REFRESH_AUTH_FAILURE_STRIKES_TO_DISABLE: u32 = 3;

/// 是否因冷却跳过刷新（`last_success` 为 Unix 秒）
pub fn should_skip_refresh_cooldown(last_success_unix: Option<u64>, now_unix: u64) -> bool {
    match last_success_unix {
        Some(ts) if now_unix.saturating_sub(ts) < REFRESH_SUCCESS_COOLDOWN_SECS => true,
        _ => false,
    }
}

/// 累加失败计数；返回是否应停用
pub fn record_refresh_auth_failure(current_strikes: u32) -> (u32, bool) {
    let next = current_strikes.saturating_add(1);
    (next, next >= REFRESH_AUTH_FAILURE_STRIKES_TO_DISABLE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::account::Account;

    fn account_with_tokens(rt: &str, at: &str, hash: Option<&str>) -> Account {
        let mut a = Account::new("a@example.com".to_string(), "t".to_string());
        a.refresh_token = Some(rt.to_string());
        a.access_token = Some(at.to_string());
        a.client_id_hash = hash.map(|s| s.to_string());
        a
    }

    fn local(rt: Option<&str>, at: Option<&str>, hash: Option<&str>) -> KiroLocalToken {
        KiroLocalToken {
            access_token: at.map(|s| s.to_string()),
            refresh_token: rt.map(|s| s.to_string()),
            expires_at: None,
            auth_method: Some("IdC".to_string()),
            provider: Some("Enterprise".to_string()),
            profile_arn: None,
            client_id_hash: hash.map(|s| s.to_string()),
            region: Some("us-east-1".to_string()),
        }
    }

    #[test]
    fn matches_by_refresh_token() {
        let acc = account_with_tokens("rt-same", "at-a", None);
        let other = account_with_tokens("rt-other", "at-b", None);
        let ide = local(Some("rt-same"), Some("at-ide"), None);
        assert!(account_matches_ide_token(&acc, &ide));
        assert!(!account_matches_ide_token(&other, &ide));
    }

    #[test]
    fn matches_by_access_prefix() {
        let acc = account_with_tokens("rt1", "abcdefghijklmnopqrstuvwxyz", None);
        let ide = local(Some("rt-diff"), Some("abcdefghijklmnopqrst"), None);
        assert!(account_matches_ide_token(&acc, &ide));
    }

    #[test]
    fn matches_by_client_id_hash() {
        let acc = account_with_tokens("rt1", "at1", Some("hash-abc"));
        let ide = local(Some("rt-x"), Some("at-y"), Some("hash-abc"));
        assert!(account_matches_ide_token(&acc, &ide));
    }

    #[test]
    fn cooldown_skips_within_window() {
        assert!(should_skip_refresh_cooldown(Some(1000), 1000 + 60));
        assert!(!should_skip_refresh_cooldown(Some(1000), 1000 + REFRESH_SUCCESS_COOLDOWN_SECS));
        assert!(!should_skip_refresh_cooldown(None, 1000));
    }

    #[test]
    fn strikes_disable_on_third() {
        assert_eq!(record_refresh_auth_failure(0), (1, false));
        assert_eq!(record_refresh_auth_failure(1), (2, false));
        assert_eq!(record_refresh_auth_failure(2), (3, true));
    }

    #[test]
    fn kam_expires_at_converts_to_rfc3339_utc() {
        let rfc = kam_expires_at_to_ide_rfc3339(Some("2026/07/20 10:02:25"));
        assert!(rfc.contains('T'), "expected RFC3339, got {rfc}");
    }
}
