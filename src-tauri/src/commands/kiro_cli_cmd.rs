#![allow(clippy::needless_pass_by_value)] // Tauri 命令需要按值传递参数

use crate::clients::kiro_client::KiroClient;
use crate::commands::common::{
    ensure_account_machine_id, extract_user_info, generate_account_machine_id,
    get_usage_by_account, get_usage_by_provider_with_machine_id,
};
use crate::core::account::Account;
use crate::kiro::cli::read_kiro_cli_accounts;
use crate::state::AppState;
use crate::utils::client_id_hash::{extract_start_url_from_client_secret, normalize_start_url};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde::Serialize;
use serde_json::Value;
use std::sync::{Mutex, MutexGuard};
use tauri::{Emitter, State};

/// 尝试从 access_token 调用 getAccount API 获取用户信息（Enterprise 账号兜底）
async fn fetch_enterprise_user_info(access_token: &str, machine_id: &str, region: &str) -> (Option<String>, Option<String>) {
    let client = match KiroClient::new() {
        Ok(c) => c,
        Err(_) => return (None, None),
    };
    // 使用公共方法调用 getAccount API
    let response = match client.get_management_api(access_token, machine_id, region, &format!("/getAccount?machineId={machine_id}")).await {
        Ok(resp) => resp,
        Err(_) => return (None, None),
    };
    let email = response.get("email").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string());
    let user_id = response.get("userId").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string())
        .or_else(|| response.get("user_id").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()));
    (email, user_id)
}

/// 从 JWT (client_secret 或 id_token) 中提取 email 和 user_id
fn extract_user_info_from_jwt(jwt: &str) -> (Option<String>, Option<String>) {
    let parts: Vec<&str> = jwt.split('.').collect();
    if parts.len() != 3 {
        return (None, None);
    }
    let payload = parts[1];
    let mut padded = payload.to_string();
    while padded.len() % 4 != 0 {
        padded.push('=');
    }
    let decoded = match URL_SAFE_NO_PAD.decode(&padded) {
        Ok(bytes) => bytes,
        Err(_) => return (None, None),
    };
    let json_str = match String::from_utf8(decoded) {
        Ok(s) => s,
        Err(_) => return (None, None),
    };
    let payload_json: Value = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(_) => return (None, None),
    };
    let email = payload_json.get("email").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string());
    let user_id = payload_json.get("user_id").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string())
        .or_else(|| payload_json.get("cognito:username").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()));
    (email, user_id)
}

/// 展开路径中的 ~ 为用户主目录
fn expand_home_dir(path: &str) -> Result<String, String> {
    if path.starts_with('~') {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .map_err(|_| "无法获取用户主目录".to_string())?;
        Ok(path.replacen('~', &home, 1))
    } else {
        Ok(path.to_string())
    }
}

/// 从 CLI 账号判断 provider
///
/// IdC 账号靠 token 自带的 start_url 区分 BuilderId 与 Enterprise（与前端 JSON 导入
/// 同源逻辑）：start_url 缺失或就是 BuilderId 默认值（view.awsapps.com/start）→ BuilderId，
/// 否则是企业自己的 d-xxx 域名 → Enterprise。这样导入的 Enterprise 账号才能保留正确
/// 的 provider，切回 IDE 时算出正确的 clientIdHash（issue #119）。
fn determine_provider(cli_account: &crate::kiro::cli::KiroCliAccount) -> String {
    if cli_account.auth_method == "social" {
        // Social Login，通过 profile_arn 判断
        if let Some(ref arn) = cli_account.profile_arn {
            if arn.contains("google") {
                return "Google".to_string();
            } else if arn.contains("github") {
                return "Github".to_string();
            }
        }
        return "Unknown".to_string();
    }

    // IdC：用 start_url 区分 BuilderId / Enterprise
    match cli_account.start_url.as_deref().map(str::trim) {
        Some(url) if !url.is_empty() && !crate::commands::common::is_builder_id_start_url(url) => {
            "Enterprise".to_string()
        }
        _ => "BuilderId".to_string(),
    }
}

/// 检查账号是否已存在
fn find_existing_account(
    accounts: &[Account],
    user_id: Option<&String>,
    _email: Option<&String>,
) -> Option<usize> {
    if let Some(uid) = user_id {
        return accounts
            .iter()
            .position(|a| a.user_id.as_ref() == Some(uid));
    }

    None
}

/// 创建账号标签
fn create_account_label(
    is_new: bool,
    token_key: &str,
    existing_account: Option<&Account>,
) -> String {
    if is_new {
        format!("从 kiro-cli 导入 ({token_key})")
    } else {
        existing_account.map_or_else(
            || format!("从 kiro-cli 导入 ({token_key})"),
            |a| a.label.clone(),
        )
    }
}

fn lock_account_store<T>(mutex: &Mutex<T>) -> Result<MutexGuard<'_, T>, String> {
    mutex
        .lock()
        .map_err(|_| "Failed to acquire store lock".to_string())
}
#[derive(Serialize)]
pub struct KiroCliImportResult {
    pub success: bool,
    pub is_new: bool,
    pub account: Option<Account>,
    pub error: Option<String>,
}

/// 获取 kiro-cli 默认数据库路径
#[tauri::command]
pub fn get_kiro_cli_default_path() -> Result<String, String> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map_err(|_| "无法获取用户主目录".to_string())?;

    let mut candidates = Vec::new();

    if cfg!(target_os = "macos") {
        candidates.push(
            std::path::PathBuf::from(&home)
                .join("Library")
                .join("Application Support")
                .join("kiro-cli")
                .join("data.sqlite3"),
        );
    } else if cfg!(target_os = "windows") {
        // Kiro CLI 2.0 原生支持 Windows
        if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
            candidates.push(
                std::path::PathBuf::from(local_app_data)
                    .join("Kiro-Cli")
                    .join("data.sqlite3"),
            );
        }
    } else {
        if let Ok(xdg_data_home) = std::env::var("XDG_DATA_HOME") {
            candidates.push(
                std::path::PathBuf::from(xdg_data_home)
                    .join("kiro-cli")
                    .join("data.sqlite3"),
            );
        }
        candidates.push(
            std::path::PathBuf::from(&home)
                .join(".local")
                .join("share")
                .join("kiro-cli")
                .join("data.sqlite3"),
        );
    }

    for path in candidates {
        if path.exists() {
            return Ok(path.to_string_lossy().to_string());
        }
    }

    // 文件不存在，返回空字符串（前端会显示占位符）
    Ok(String::new())
}
/// 从 kiro-cli 数据库导入账号
#[tauri::command]
pub async fn import_from_kiro_cli(
    db_path: String,
    state: State<'_, AppState>,
) -> Result<KiroCliImportResult, String> {
    eprintln!("[Kiro CLI Import] 开始导入，数据库路径: {db_path}");

    // 展开 ~ 为用户主目录
    let expanded_path = expand_home_dir(&db_path)?;
    eprintln!("[Kiro CLI Import] 展开后的路径: {expanded_path}");

    // 1. 读取 kiro-cli 数据库
    let cli_accounts = read_kiro_cli_accounts(&expanded_path)?;

    if cli_accounts.is_empty() {
        return Err("数据库中没有账号数据".to_string());
    }

    if cli_accounts.len() > 1 {
        return Err("数据库中有多个账号，请联系开发者".to_string());
    }

    let cli_account = &cli_accounts[0];
    let auth_method = &cli_account.auth_method;
    let token_key = &cli_account.token_key;
    eprintln!("[Kiro CLI Import] 读取到账号: auth_method={auth_method}, token_key={token_key}");

    // 2. 调用统一的 getUsageLimits API 获取配额
    let provider = determine_provider(cli_account);
    let account_machine_id = {
        let store = lock_account_store(&state.store)?;
        store
            .accounts
            .iter()
            .find(|account| account.refresh_token.as_ref() == Some(&cli_account.refresh_token))
            .and_then(|account| {
                account
                    .machine_id
                    .clone()
                    .filter(|id| !id.trim().is_empty())
            })
            .unwrap_or_else(generate_account_machine_id)
    };
    let usage_result = get_usage_by_provider_with_machine_id(
        &provider,
        &cli_account.access_token,
        &account_machine_id,
    )
    .await;

    let (email, user_id, usage_data, is_banned, is_auth_error) = match usage_result {
        Ok(result) => {
            let (email, user_id) = extract_user_info(&result.usage_data);
            let (email, user_id) = if provider == "Enterprise" && email.is_none() && user_id.is_none() {
                eprintln!("[Kiro CLI Import] Enterprise account: getUsageLimits returned no userInfo, trying fallback APIs");
                // 尝试从 client_secret JWT 解析
                let (email_from_secret, user_id_from_secret) = cli_account
                    .client_secret
                    .as_ref()
                    .and_then(|cs| Some(extract_user_info_from_jwt(cs)))
                    .unwrap_or((None, None));
                if email_from_secret.is_some() || user_id_from_secret.is_some() {
                    eprintln!("[Kiro CLI Import] Decoded from client_secret - email: {:?}, user_id: {:?}", email_from_secret, user_id_from_secret);
                    (email_from_secret, user_id_from_secret)
                } else {
                    // 尝试调用 getAccount API
                    let (email_from_api, user_id_from_api) = fetch_enterprise_user_info(
                        &cli_account.access_token,
                        &account_machine_id,
                        &cli_account.region,
                    ).await;
                    if email_from_api.is_some() || user_id_from_api.is_some() {
                        eprintln!("[Kiro CLI Import] Fetched from getAccount API - email: {:?}, user_id: {:?}", email_from_api, user_id_from_api);
                        (email_from_api, user_id_from_api)
                    } else {
                        (email, user_id)
                    }
                }
            } else {
                (email, user_id)
            };
            (
                email,
                user_id,
                Some(result.usage_data),
                result.is_banned,
                result.is_auth_error,
            )
        }
        Err(e) => {
            eprintln!("[Kiro CLI Import] 获取配额失败: {e}");
            return Ok(KiroCliImportResult {
                success: false,
                is_new: false,
                account: None,
                error: Some(format!("获取账号信息失败: {e}")),
            });
        }
    };

    // 3. 检查账号是否已存在
    let mut store = lock_account_store(&state.store)?;
    let existing_index = find_existing_account(&store.accounts, user_id.as_ref(), email.as_ref());
    let is_new = existing_index.is_none();

    // 4. 创建或更新 Account
    let existing_account = existing_index.and_then(|idx| store.accounts.get(idx));
    let label = create_account_label(is_new, &cli_account.token_key, existing_account);

    let mut account = if let Some(e) = email.clone() {
        Account::new(e, label)
    } else if let Some(uid) = user_id.clone() {
        Account::new_enterprise(uid, label)
    } else {
        return Ok(KiroCliImportResult {
            success: false,
            is_new: false,
            account: None,
            error: Some("无法获取账号标识（email 或 userId）".to_string()),
        });
    };

    // 5. 填充字段
    account.access_token = Some(cli_account.access_token.clone());
    account.refresh_token = Some(cli_account.refresh_token.clone());
    account.expires_at.clone_from(&cli_account.expires_at);
    account.provider = Some(provider.clone());
    account.user_id = user_id;
    account.region = Some(cli_account.region.clone());
    account.usage_data = usage_data;

    // 更新账号状态（包括封禁检测）
    crate::commands::common::update_account_status(&mut account, is_banned, is_auth_error);

    // 6. 根据认证类型填充字段
    if cli_account.auth_method == "social" {
        account.auth_method = Some("social".to_string());
        account.profile_arn.clone_from(&cli_account.profile_arn);
    } else {
        account.auth_method = Some("IdC".to_string());
        account.client_id.clone_from(&cli_account.client_id);
        account.client_secret.clone_from(&cli_account.client_secret);

        // start_url：优先用 token 自带的，缺失则回退到 clientSecret JWT（与添加路径同源）。
        // 切回 IDE 时要靠它算出正确的 clientIdHash —— Enterprise 必须是自己的 d-xxx 域名。
        // 统一 normalize_start_url 去尾斜杠（JWT 那条已在真相源规范化，token 那条这里兜底），
        // 保证落进 account.start_url 的永远是无斜杠规范形。
        let start_url = cli_account
            .start_url
            .clone()
            .filter(|s| !s.trim().is_empty())
            .or_else(|| {
                cli_account
                    .client_secret
                    .as_deref()
                    .and_then(extract_start_url_from_client_secret)
            })
            .map(|s| normalize_start_url(&s));

        // clientIdHash：走统一裁决点，Enterprise 在此硬校验（issue #119）。导入数据被污染
        // （Enterprise 缺 start_url / 落到 BuilderId 默认值）时直接返回结构化失败，不写坏数据。
        let client_id_hash = match crate::commands::common::resolve_idc_client_id_hash(
            &provider,
            None,
            start_url.as_deref(),
        ) {
            Ok(hash) => hash,
            Err(e) => {
                return Ok(KiroCliImportResult {
                    success: false,
                    is_new: false,
                    account: None,
                    error: Some(format!("解析 clientIdHash 失败: {e}")),
                });
            }
        };
        account.client_id_hash = Some(client_id_hash);
        account.start_url = start_url;
    }

    // 7. 生成或保留 machine_id
    if let Some(idx) = existing_index {
        // 更新现有账号，保留 machine_id；历史空值则回填本次导入使用的账号级 ID
        account
            .machine_id
            .clone_from(&store.accounts[idx].machine_id);
        if account
            .machine_id
            .as_ref()
            .is_none_or(|id| id.trim().is_empty())
        {
            account.machine_id = Some(account_machine_id);
        }
        account.id.clone_from(&store.accounts[idx].id);
        store.accounts[idx] = account.clone();
    } else {
        // 新账号，保存本次 usage 检测使用的账号级 machine_id
        account.machine_id = Some(account_machine_id);
        store.accounts.push(account.clone());
    }

    store.save_to_file();
    drop(store);

    let email = &account.email;
    let user_id = &account.user_id;
    eprintln!("[Kiro CLI Import] 导入成功: is_new={is_new}, email={email:?}, user_id={user_id:?}");

    Ok(KiroCliImportResult {
        success: true,
        is_new,
        account: Some(account),
        error: None,
    })
}

// ============================================================
// CLI 2.0 切号功能
// ============================================================

/// 检测 CLI 2.0 安装状态
#[tauri::command]
pub fn check_cli_installation() -> crate::kiro::cli::CliInstallationInfo {
    crate::kiro::cli::check_cli_installation()
}

/// 读取 CLI 数据库快照（前端展示用）
#[tauri::command]
pub fn read_cli_db_snapshot(
    db_path: String,
) -> Result<crate::kiro::cli::KiroCliDbSnapshot, String> {
    let expanded_path = expand_home_dir(&db_path)?;
    crate::kiro::cli::read_cli_db_snapshot(&expanded_path)
}

/// 切号到 CLI 账号
#[tauri::command]
pub async fn switch_to_cli_account(
    account_id: String,
    db_path: String,
    state: State<'_, AppState>,
    app: tauri::AppHandle,
) -> Result<crate::kiro::cli::KiroCliWriteBackup, String> {
    let expanded_path = expand_home_dir(&db_path)?;

    // 1. 从 store 读取账号数据
    let account = {
        let store = lock_account_store(&state.store)?;
        store
            .accounts
            .iter()
            .find(|a| a.id == account_id)
            .cloned()
            .ok_or_else(|| format!("账号不存在: {account_id}"))?
    };

    // 2. 切号前刷新 token（确保写入的是有效 token）
    let refreshed_account = if account.refresh_token.is_some() {
        log::info!("[CLI Switch] 切号前刷新 token...");
        match crate::commands::common::refresh_token_by_provider(&account).await {
            Ok(refresh_result) => {
                log::info!("[CLI Switch] Token 刷新成功");
                // 更新 store 中的 token
                {
                    let mut store = lock_account_store(&state.store)?;
                    if let Some(a) = store.accounts.iter_mut().find(|a| a.id == account_id) {
                        crate::commands::common::apply_refreshed_account_tokens(a, &refresh_result);
                        let _ = crate::commands::common::save_store(&store);
                    }
                }
                // 构造刷新后的账号对象
                let mut updated = account.clone();
                updated.access_token = Some(refresh_result.access_token);
                updated.refresh_token = refresh_result.refresh_token;
                // 根据 expires_in 计算 expires_at
                let expires_at =
                    chrono::Utc::now() + chrono::Duration::seconds(refresh_result.expires_in);
                updated.expires_at = Some(expires_at.to_rfc3339());
                updated
            }
            Err(e) => {
                log::warn!("[CLI Switch] Token 刷新失败: {}, 使用现有 token", e);
                account
            }
        }
    } else {
        account
    };

    let mut refreshed_account = refreshed_account;
    let generated_machine_id = if refreshed_account
        .machine_id
        .as_ref()
        .is_none_or(|id| id.trim().is_empty())
    {
        Some(ensure_account_machine_id(&mut refreshed_account))
    } else {
        None
    };
    if let Some(machine_id) = generated_machine_id {
        let mut store = lock_account_store(&state.store)?;
        if let Some(a) = store.accounts.iter_mut().find(|a| a.id == account_id) {
            if a.machine_id.as_ref().is_none_or(|id| id.trim().is_empty()) {
                a.machine_id = Some(machine_id);
                let _ = crate::commands::common::save_store(&store);
            }
        }
    }

    // 3. 切号后立即获取配额检测封禁状态
    if refreshed_account.provider.is_none() {
        return Err("账号缺少 provider 字段".to_string());
    }
    let access_token = refreshed_account
        .access_token
        .as_ref()
        .ok_or("账号缺少 access_token")?;

    log::info!("[CLI Switch] 切号后检测账号状态...");
    match get_usage_by_account(&refreshed_account, access_token).await {
        Ok(usage_result) => {
            // 更新账号状态（包括封禁检测）
            let mut store = lock_account_store(&state.store)?;
            if let Some(a) = store.accounts.iter_mut().find(|a| a.id == account_id) {
                a.usage_data = Some(usage_result.usage_data);
                crate::commands::common::update_account_status(
                    a,
                    usage_result.is_banned,
                    usage_result.is_auth_error,
                );
                let _ = crate::commands::common::save_store(&store);

                // 通知前端刷新账号列表
                let _ = app.emit("accounts-updated", ());

                if usage_result.is_banned {
                    log::warn!("[CLI Switch] 检测到账号已封禁");
                    return Err("账号已被封禁，无法切换到 CLI".to_string());
                }
            }
        }
        Err(e) => {
            log::warn!("[CLI Switch] 获取配额失败: {}, 继续切号", e);
            // 获取配额失败不阻止切号，但记录警告
        }
    }

    // 4. 构造切号载荷
    let payload = crate::kiro::cli::build_cli_switch_payload(&refreshed_account)?;

    // 5. 执行切号写入（包括清除旧 key）
    crate::kiro::cli::switch_cli_account(&expanded_path, &payload)
}

/// 回滚切号操作
#[tauri::command]
pub fn rollback_cli_switch(
    db_path: String,
    backup: crate::kiro::cli::KiroCliWriteBackup,
) -> Result<(), String> {
    let expanded_path = expand_home_dir(&db_path)?;
    crate::kiro::cli::rollback_cli_switch(&expanded_path, &backup)
}

/// 退出 CLI 2.0 当前登录态（清空所有 token key，切号的逆操作）。
///
/// 数据库不存在时视为本来就没登录，幂等返回 0。返回实际清掉的 token key 数量。
#[tauri::command]
pub fn logout_cli_account(db_path: String) -> Result<usize, String> {
    let expanded_path = expand_home_dir(&db_path)?;
    // 数据库文件不存在 = 没装/没登录 CLI，幂等返回 0，不报错
    if !std::path::Path::new(&expanded_path).exists() {
        return Ok(0);
    }
    crate::kiro::cli::logout_cli_account(&expanded_path)
}

#[cfg(test)]
mod tests {
    use super::lock_account_store;
    use std::sync::Mutex;

    #[test]
    fn lock_account_store_returns_error_when_mutex_is_poisoned() {
        let mutex = Mutex::new(());
        let _ = std::panic::catch_unwind(|| {
            let _guard = mutex.lock().expect("mutex should lock before poison");
            panic!("poison lock");
        });

        let err = lock_account_store(&mutex).expect_err("poisoned mutex should return error");
        assert!(err.contains("store lock"));
    }
}
