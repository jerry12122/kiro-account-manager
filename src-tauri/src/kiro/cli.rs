use crate::core::account::Account;
use crate::utils::client_id_hash::normalize_start_url;
use rusqlite::{Connection, Result as SqliteResult};
use serde::{Deserialize, Serialize};
use std::path::Path;

// ============================================================
// CLI 2.0 固定常量（基于实测数据库样本）
// ============================================================

/// CLI 2.0 OIDC 固定 scopes（不可变）
const CLI_OIDC_SCOPES: &[&str] = &[
    "codewhisperer:completions",
    "codewhisperer:analysis",
    "codewhisperer:conversations",
];

/// CLI 2.0 OIDC 固定 oauth_flow
const CLI_OAUTH_FLOW: &str = "Pkce";

/// CLI 2.0 Social 登录固定 start_url（如果源账号没有）
const CLI_SOCIAL_START_URL: &str = "https://view.awsapps.com/start";

/// CLI 2.0 默认 region
const CLI_DEFAULT_REGION: &str = "us-east-1";

/// Kiro CLI 账号数据
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct KiroCliAccount {
    pub access_token: String,
    pub refresh_token: String,
    pub profile_arn: Option<String>,
    pub region: String,
    pub expires_at: Option<String>,
    pub scopes: Option<Vec<String>>,
    pub auth_method: String, // "social" 或 "IdC"
    pub token_key: String,   // 记录来源键名
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    /// IdC/SSO token 自带的 start_url（真实 kiro-cli token 里就有，
    /// 用来区分 BuilderId 与 Enterprise，并还原正确的 clientIdHash）
    pub start_url: Option<String>,
}

/// Device Registration 数据（仅 AWS SSO OIDC）
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DeviceRegistration {
    pub client_id: String,
    pub client_secret: String,
    pub region: String,
}

/// CLI 数据库完整快照（用于读取当前状态）
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct KiroCliDbSnapshot {
    pub token_entries: Vec<KiroCliAuthEntry>,
    pub device_registration: Option<DeviceRegistration>,
    pub db_path: String,
}

/// CLI 认证条目（从 auth_kv 读取的原始记录）
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct KiroCliAuthEntry {
    pub key: String,
    pub value_json: String,
    pub parsed_token: Option<TokenData>,
}

/// Token 数据（解析后的结构）
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TokenData {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_at: Option<String>,
    pub region: String,
    pub start_url: Option<String>,
    pub oauth_flow: Option<String>,
    pub scopes: Option<Vec<String>>,
}

/// CLI 切号写入载荷（准备写入 DB 的目标记录）
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct KiroCliSwitchPayload {
    pub token_key: String,
    pub token_value: String,
    pub device_reg_key: String,
    pub device_reg_value: String,
}

/// 写入前的备份数据（用于回滚）
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct KiroCliWriteBackup {
    pub old_token: Option<(String, String)>,
    pub old_device_reg: Option<(String, String)>,
    pub backup_time: String,
}

/// 从 kiro-cli 数据库读取账号
pub fn read_kiro_cli_accounts(db_path: &str) -> Result<Vec<KiroCliAccount>, String> {
    // 检查文件是否存在
    if !Path::new(db_path).exists() {
        return Err(format!("数据库文件不存在: {db_path}"));
    }

    // 打开数据库（只读模式）
    let conn = Connection::open_with_flags(db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| format!("无法打开数据库: {e}"))?;

    let mut accounts = Vec::new();

    // 按优先级尝试读取 Token
    let token_keys = vec![
        "kirocli:social:token",
        "kirocli:odic:token",
        "codewhisperer:odic:token",
    ];

    for key in token_keys {
        if let Ok(mut account) = read_token_from_db(&conn, key) {
            // 如果是 IdC，尝试读取 Device Registration
            if account.auth_method == "IdC" {
                if let Ok(device_reg) = read_device_registration(&conn) {
                    account.client_id = Some(device_reg.client_id);
                    account.client_secret = Some(device_reg.client_secret);
                }
            }
            accounts.push(account);
            break; // 只导入第一个找到的账号
        }
    }

    if accounts.is_empty() {
        return Err("未找到有效的账号数据".to_string());
    }

    Ok(accounts)
}

/// 从数据库读取指定键的 Token
fn read_token_from_db(conn: &Connection, key: &str) -> SqliteResult<KiroCliAccount> {
    let mut stmt = conn.prepare("SELECT value FROM auth_kv WHERE key = ?1")?;
    let value: String = stmt.query_row([key], |row| row.get(0))?;

    // 解析 JSON
    let token_data: serde_json::Value =
        serde_json::from_str(&value).map_err(|_| rusqlite::Error::InvalidQuery)?;

    // 提取字段
    let access_token = token_data["access_token"]
        .as_str()
        .ok_or(rusqlite::Error::InvalidQuery)?
        .to_string();

    let refresh_token = token_data["refresh_token"]
        .as_str()
        .ok_or(rusqlite::Error::InvalidQuery)?
        .to_string();

    let region = token_data["region"]
        .as_str()
        .unwrap_or("us-east-1")
        .to_string();

    let expires_at = token_data["expires_at"]
        .as_str()
        .map(std::string::ToString::to_string);

    let profile_arn = token_data["profile_arn"]
        .as_str()
        .map(std::string::ToString::to_string);

    let start_url = token_data["start_url"]
        .as_str()
        .map(std::string::ToString::to_string);

    let scopes = token_data["scopes"].as_array().map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_str().map(std::string::ToString::to_string))
            .collect()
    });

    // 判断认证类型
    let auth_method = if profile_arn.is_some() {
        "social".to_string()
    } else if scopes.is_some() {
        "IdC".to_string()
    } else {
        "unknown".to_string()
    };

    Ok(KiroCliAccount {
        access_token,
        refresh_token,
        profile_arn,
        region,
        expires_at,
        scopes,
        auth_method,
        token_key: key.to_string(),
        client_id: None,
        client_secret: None,
        start_url,
    })
}

/// 读取 Device Registration（OIDC 专用）
fn read_device_registration(conn: &Connection) -> SqliteResult<DeviceRegistration> {
    // 按优先级尝试读取
    let keys = vec![
        "kirocli:odic:device-registration",
        "codewhisperer:odic:device-registration",
    ];

    for key in keys {
        if let Ok(device_reg) = read_device_registration_by_key(conn, key) {
            return Ok(device_reg);
        }
    }

    Err(rusqlite::Error::QueryReturnedNoRows)
}

/// 从数据库读取指定键的 Device Registration
fn read_device_registration_by_key(
    conn: &Connection,
    key: &str,
) -> SqliteResult<DeviceRegistration> {
    let mut stmt = conn.prepare("SELECT value FROM auth_kv WHERE key = ?1")?;
    let value: String = stmt.query_row([key], |row| row.get(0))?;

    // 解析 JSON
    let data: serde_json::Value =
        serde_json::from_str(&value).map_err(|_| rusqlite::Error::InvalidQuery)?;

    let client_id = data["client_id"]
        .as_str()
        .ok_or(rusqlite::Error::InvalidQuery)?
        .to_string();

    let client_secret = data["client_secret"]
        .as_str()
        .ok_or(rusqlite::Error::InvalidQuery)?
        .to_string();

    let region = data["region"].as_str().unwrap_or("us-east-1").to_string();

    Ok(DeviceRegistration {
        client_id,
        client_secret,
        region,
    })
}

/// 规范化 Token 数据，补齐 CLI 2.0 固定字段
/// 用于切号写入前确保字段完整性
#[allow(dead_code)]
fn normalize_token_for_cli(token: &mut TokenData, auth_method: &str) {
    // IdC 账号：强制补齐 scopes 和 oauth_flow
    if auth_method == "IdC" {
        if token.scopes.is_none() {
            token.scopes = Some(CLI_OIDC_SCOPES.iter().map(|s| s.to_string()).collect());
        }
        if token.oauth_flow.is_none() {
            token.oauth_flow = Some(CLI_OAUTH_FLOW.to_string());
        }
    }

    // Social 账号：补齐 start_url（如果缺失）
    if auth_method == "social" && token.start_url.is_none() {
        token.start_url = Some(CLI_SOCIAL_START_URL.to_string());
    }

    // 所有账号：补齐 region（如果缺失）
    // 所有账号:补齐 region(如果缺失)
    if token.region.is_empty() {
        token.region = CLI_DEFAULT_REGION.to_string();
    }
}

/// 切号写入 CLI 2.0 数据库
pub fn switch_cli_account(
    db_path: &str,
    payload: &KiroCliSwitchPayload,
) -> Result<KiroCliWriteBackup, String> {
    if !Path::new(db_path).exists() {
        return Err(format!("数据库文件不存在: {db_path}"));
    }

    let mut conn = Connection::open(db_path).map_err(|e| format!("无法打开数据库: {e}"))?;

    // 开启事务
    let tx = conn
        .transaction()
        .map_err(|e| format!("无法开启事务: {e}"))?;

    // 备份旧值
    let old_token = read_kv_value(&tx, &payload.token_key).ok();
    let old_device_reg = read_kv_value(&tx, &payload.device_reg_key).ok();

    // 写入新值
    write_kv_value(&tx, &payload.token_key, &payload.token_value)
        .map_err(|e| format!("写入 token 失败: {e}"))?;
    write_kv_value(&tx, &payload.device_reg_key, &payload.device_reg_value)
        .map_err(|e| format!("写入 device registration 失败: {e}"))?;

    // 清除其他优先级的旧 token key（与 Electron 版本一致）
    let all_token_keys = vec![
        "kirocli:social:token",
        "kirocli:odic:token",
        "codewhisperer:odic:token",
    ];
    for key in all_token_keys {
        if key != payload.token_key {
            // 忽略删除失败（key 可能不存在）
            let _ = tx.execute("DELETE FROM auth_kv WHERE key = ?1", [key]);
        }
    }

    // 提交事务
    tx.commit().map_err(|e| format!("提交事务失败: {e}"))?;

    Ok(KiroCliWriteBackup {
        old_token: old_token.map(|v| (payload.token_key.clone(), v)),
        old_device_reg: old_device_reg.map(|v| (payload.device_reg_key.clone(), v)),
        backup_time: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
    })
}

/// 退出 CLI 2.0 当前登录态：清空所有 token key（登录的逆操作）。
///
/// 与 `switch_cli_account` 对称——切号是写入某个 token key，退出登录则删除全部 token key，
/// 让 CLI 回到"未登录"状态。device-registration 保留（那是设备注册信息，非登录凭证，
/// 重新登录时复用）。返回删除的 key 数量，便于前端区分"本来就没登录"与"清掉了登录态"。
pub fn logout_cli_account(db_path: &str) -> Result<usize, String> {
    if !Path::new(db_path).exists() {
        return Err(format!("数据库文件不存在: {db_path}"));
    }

    let mut conn = Connection::open(db_path).map_err(|e| format!("无法打开数据库: {e}"))?;

    let tx = conn
        .transaction()
        .map_err(|e| format!("无法开启事务: {e}"))?;

    // 清空所有优先级的 token key（与 switch_cli_account 清理的集合一致）
    let all_token_keys = [
        "kirocli:social:token",
        "kirocli:odic:token",
        "codewhisperer:odic:token",
    ];
    let mut removed = 0usize;
    for key in all_token_keys {
        // 忽略单条删除失败（key 可能不存在），用 changes 累计实际删除数量
        if let Ok(n) = tx.execute("DELETE FROM auth_kv WHERE key = ?1", [key]) {
            removed += n;
        }
    }

    tx.commit().map_err(|e| format!("提交事务失败: {e}"))?;
    Ok(removed)
}

/// 读取 auth_kv 键值
fn read_kv_value(conn: &Connection, key: &str) -> SqliteResult<String> {
    let mut stmt = conn.prepare("SELECT value FROM auth_kv WHERE key = ?1")?;
    stmt.query_row([key], |row| row.get(0))
}

/// 写入 auth_kv 键值(INSERT OR REPLACE)
fn write_kv_value(conn: &Connection, key: &str, value: &str) -> SqliteResult<()> {
    conn.execute(
        "INSERT OR REPLACE INTO auth_kv (key, value) VALUES (?1, ?2)",
        [key, value],
    )?;
    Ok(())
}

/// 读取 CLI 数据库完整快照（用于前端展示）
pub fn read_cli_db_snapshot(db_path: &str) -> Result<KiroCliDbSnapshot, String> {
    if !Path::new(db_path).exists() {
        return Err(format!("数据库文件不存在: {db_path}"));
    }

    let conn = Connection::open_with_flags(db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| format!("无法打开数据库: {e}"))?;

    // 读取所有 token 条目
    let mut stmt = conn
        .prepare("SELECT key, value FROM auth_kv WHERE key LIKE '%token'")
        .map_err(|e| format!("查询失败: {e}"))?;

    let entries: Vec<KiroCliAuthEntry> = stmt
        .query_map([], |row| {
            let key: String = row.get(0)?;
            let value_json: String = row.get(1)?;
            let parsed_token = parse_token_data(&value_json).ok();
            Ok(KiroCliAuthEntry {
                key,
                value_json,
                parsed_token,
            })
        })
        .map_err(|e| format!("读取条目失败: {e}"))?
        .filter_map(Result::ok)
        .collect();

    // 读取 device registration
    let device_registration = read_device_registration(&conn).ok();

    Ok(KiroCliDbSnapshot {
        token_entries: entries,
        device_registration,
        db_path: db_path.to_string(),
    })
}

/// 解析 Token JSON 数据
fn parse_token_data(json_str: &str) -> Result<TokenData, String> {
    let data: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("JSON 解析失败: {e}"))?;

    Ok(TokenData {
        access_token: data["access_token"].as_str().unwrap_or("").to_string(),
        refresh_token: data["refresh_token"].as_str().unwrap_or("").to_string(),
        expires_at: data["expires_at"].as_str().map(String::from),
        region: data["region"]
            .as_str()
            .unwrap_or(CLI_DEFAULT_REGION)
            .to_string(),
        start_url: data["start_url"].as_str().map(String::from),
        oauth_flow: data["oauth_flow"].as_str().map(String::from),
        scopes: data["scopes"].as_array().map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        }),
    })
}

/// 回滚切号操作（恢复备份数据）
pub fn rollback_cli_switch(db_path: &str, backup: &KiroCliWriteBackup) -> Result<(), String> {
    if !Path::new(db_path).exists() {
        return Err(format!("数据库文件不存在: {db_path}"));
    }

    let mut conn = Connection::open(db_path).map_err(|e| format!("无法打开数据库: {e}"))?;

    let tx = conn
        .transaction()
        .map_err(|e| format!("无法开启事务: {e}"))?;

    // 恢复 token
    if let Some((key, value)) = &backup.old_token {
        write_kv_value(&tx, key, value).map_err(|e| format!("恢复 token 失败: {e}"))?;
    }

    // 恢复 device registration
    if let Some((key, value)) = &backup.old_device_reg {
        write_kv_value(&tx, key, value)
            .map_err(|e| format!("恢复 device registration 失败: {e}"))?;
    }

    tx.commit().map_err(|e| format!("提交回滚事务失败: {e}"))?;

    Ok(())
}

// ============================================================
// CLI 2.0 环境检测
// ============================================================

/// CLI 安装检测结果
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CliInstallationInfo {
    pub cli_installed: bool,
    pub cli_path: Option<String>,
    pub db_path: Option<String>,
    pub db_exists: bool,
}

/// 检测 CLI 2.0 是否安装
pub fn check_cli_installation() -> CliInstallationInfo {
    let cli_path = detect_cli_executable();
    let db_path = detect_cli_database();

    let cli_installed = cli_path.is_some();
    let db_exists = db_path
        .as_ref()
        .is_some_and(|p| std::path::Path::new(p).exists());

    CliInstallationInfo {
        cli_installed,
        cli_path,
        db_path,
        db_exists,
    }
}

/// 检测 CLI 可执行文件
pub fn detect_cli_executable() -> Option<String> {
    let candidates = get_cli_executable_paths();

    for path in candidates {
        if path.exists() {
            return Some(path.to_string_lossy().to_string());
        }
    }

    None
}

/// 获取 CLI 可执行文件候选路径
fn get_cli_executable_paths() -> Vec<std::path::PathBuf> {
    let mut paths = Vec::new();

    if cfg!(target_os = "windows") {
        if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
            paths.push(
                std::path::PathBuf::from(local_app_data)
                    .join("Kiro-Cli")
                    .join("kiro-cli.exe"),
            );
        }
    } else if cfg!(target_os = "macos") {
        if let Ok(home) = std::env::var("HOME") {
            // macOS 可能的安装位置
            paths.push(std::path::PathBuf::from("/usr/local/bin/kiro-cli"));
            paths.push(
                std::path::PathBuf::from(&home)
                    .join("Library")
                    .join("Application Support")
                    .join("kiro-cli")
                    .join("bin")
                    .join("kiro-cli"),
            );
        }
    } else {
        // Linux
        if let Ok(home) = std::env::var("HOME") {
            paths.push(std::path::PathBuf::from("/usr/local/bin/kiro-cli"));
            paths.push(std::path::PathBuf::from(&home).join(".local/bin/kiro-cli"));
        }
    }

    paths
}

/// 检测 CLI 数据库
pub fn detect_cli_database() -> Option<String> {
    let candidates = get_cli_database_paths();

    for path in &candidates {
        if path.exists() {
            return Some(path.to_string_lossy().to_string());
        }
    }

    // 返回默认路径（即使不存在）
    candidates.first().map(|p| p.to_string_lossy().to_string())
}

/// 仅在数据库文件已存在时返回路径（不回退到默认占位路径）
pub fn detect_existing_cli_database() -> Option<String> {
    get_cli_database_paths()
        .into_iter()
        .find(|path| path.exists())
        .map(|path| path.to_string_lossy().to_string())
}

/// CLI token 的 expires_at：优先用账号已有过期时间，否则 now+1h。
/// 真实 kiro-cli 用 RFC3339 + `Z` + 微秒（而非 chrono 默认的 `+00:00`）。
fn cli_token_expires_at(account: &Account) -> String {
    use chrono::TimeZone;

    if let Some(raw) = account
        .expires_at
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(raw, "%Y/%m/%d %H:%M:%S") {
            if let Some(local) = chrono::Local.from_local_datetime(&naive).single() {
                return local
                    .with_timezone(&chrono::Utc)
                    .to_rfc3339_opts(chrono::SecondsFormat::Micros, true);
            }
        }
        if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(raw) {
            return dt
                .with_timezone(&chrono::Utc)
                .to_rfc3339_opts(chrono::SecondsFormat::Micros, true);
        }
    }

    (chrono::Utc::now() + chrono::Duration::hours(1))
        .to_rfc3339_opts(chrono::SecondsFormat::Micros, true)
}

/// 构造切号载荷（从 Account 转换为 CLI 2.0 格式）
pub fn build_cli_switch_payload(account: &Account) -> Result<KiroCliSwitchPayload, String> {
    let provider = account.provider.as_ref().ok_or("账号缺少 provider 字段")?;
    let (token_key, device_reg_key, auth_method) = match provider.as_str() {
        // Enterprise 也是 IdC/SSO，写入 odic key（真实 kiro-cli 实测样本即来自 SSO 登录）
        "BuilderId" | "Enterprise" => (
            "kirocli:odic:token",
            "kirocli:odic:device-registration",
            "IdC",
        ),
        "Google" | "Github" => (
            "kirocli:social:token",
            "kirocli:social:device-registration",
            "social",
        ),
        _ => return Err(format!("不支持的 provider: {provider}")),
    };

    // Social 默认 profile_arn（与 Electron 版本一致；IdC token 不带 profile_arn）
    const SOCIAL_PROFILE_ARN: &str =
        "arn:aws:codewhisperer:us-east-1:699475941385:profile/EHGA3GRVQMUK";
    // CLI 默认 Builder ID start_url（Enterprise 用账号自带的 d-xxx start_url）
    const DEFAULT_START_URL: &str = crate::commands::common::KIRO_BUILDER_ID_START_URL;

    let default_region = "us-east-1".to_string();
    let region = account.region.as_ref().unwrap_or(&default_region);
    let token_expires_at = cli_token_expires_at(account);

    let mut token_data = serde_json::json!({
        "access_token": account.access_token,
        "refresh_token": account.refresh_token,
        "expires_at": token_expires_at,
        "region": region,
        "oauth_flow": "Pkce",
        "scopes": [
            "codewhisperer:completions",
            "codewhisperer:analysis",
            "codewhisperer:conversations"
        ],
    });

    if auth_method == "IdC" {
        // IdC/SSO token：带 start_url，且 **不带** profile_arn（与真实 kiro-cli 一致）
        let start_url = match provider.as_str() {
            "BuilderId" => account
                .start_url
                .as_deref()
                .map(normalize_start_url)
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| DEFAULT_START_URL.to_string()),
            // Enterprise 必须用账号自带的 d-xxx 域名，绝不能回退到 BuilderId 默认值
            "Enterprise" => {
                let url = account
                    .start_url
                    .as_deref()
                    .map(normalize_start_url)
                    .filter(|s| !s.is_empty())
                    .ok_or(
                        "Enterprise 账号必须提供 start_url（企业自己的 d-xxx 域名），\
                         不能为空",
                    )?;
                if crate::commands::common::is_builder_id_start_url(&url) {
                    return Err(
                        "Enterprise 账号的 start_url 不能是 BuilderId 默认值\
                         （https://view.awsapps.com/start），请填入企业自己的 d-xxx 域名"
                            .to_string(),
                    );
                }
                url
            }
            _ => return Err(format!("不支持的 IdC provider: {provider}")),
        };
        token_data["start_url"] = serde_json::json!(start_url);
    } else {
        token_data["start_url"] = serde_json::json!(DEFAULT_START_URL);
        let profile_arn = account
            .profile_arn
            .as_deref()
            .filter(|s| !s.is_empty())
            .unwrap_or(SOCIAL_PROFILE_ARN);
        token_data["profile_arn"] = serde_json::json!(profile_arn);
    }

    let token_value =
        serde_json::to_string(&token_data).map_err(|e| format!("序列化 token 失败: {e}"))?;

    // device-registration：本地未持久化 client_secret 过期时间，按 90 天兜底
    let secret_expires_at = (chrono::Utc::now() + chrono::Duration::days(90))
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let empty = String::new();
    let device_reg_data = serde_json::json!({
        "client_id": account.client_id.as_ref().unwrap_or(&empty),
        "client_secret": account.client_secret.as_ref().unwrap_or(&empty),
        "client_secret_expires_at": secret_expires_at,
        "region": region,
        "oauth_flow": "Pkce",
        "scopes": [
            "codewhisperer:completions",
            "codewhisperer:analysis",
            "codewhisperer:conversations"
        ],
    });

    let device_reg_value = serde_json::to_string(&device_reg_data)
        .map_err(|e| format!("序列化 device registration 失败: {e}"))?;

    Ok(KiroCliSwitchPayload {
        token_key: token_key.to_string(),
        token_value,
        device_reg_key: device_reg_key.to_string(),
        device_reg_value,
    })
}

/// 把 KAM 账号凭证写入本地 CLI（access/refresh/expires_at + device-registration）。
///
/// 数据库不存在时返回 `Ok(false)`（未安装 CLI 不算错误）；写入成功返回 `Ok(true)`。
pub fn sync_account_to_cli(account: &Account) -> Result<bool, String> {
    let Some(db_path) = detect_existing_cli_database() else {
        return Ok(false);
    };
    let payload = build_cli_switch_payload(account)?;
    switch_cli_account(&db_path, &payload)?;
    Ok(true)
}

/// 获取 CLI 数据库候选路径
fn get_cli_database_paths() -> Vec<std::path::PathBuf> {
    let mut paths = Vec::new();

    if cfg!(target_os = "windows") {
        if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
            paths.push(
                std::path::PathBuf::from(local_app_data)
                    .join("Kiro-Cli")
                    .join("data.sqlite3"),
            );
        }
    } else if cfg!(target_os = "macos") {
        if let Ok(home) = std::env::var("HOME") {
            paths.push(
                std::path::PathBuf::from(&home)
                    .join("Library")
                    .join("Application Support")
                    .join("kiro-cli")
                    .join("data.sqlite3"),
            );
        }
    } else {
        // Linux
        if let Ok(home) = std::env::var("HOME") {
            if let Ok(xdg_data_home) = std::env::var("XDG_DATA_HOME") {
                paths.push(
                    std::path::PathBuf::from(xdg_data_home)
                        .join("kiro-cli")
                        .join("data.sqlite3"),
                );
            }
            paths.push(
                std::path::PathBuf::from(&home)
                    .join(".local")
                    .join("share")
                    .join("kiro-cli")
                    .join("data.sqlite3"),
            );
        }
    }

    paths
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 在系统临时目录建一个带 auth_kv 表的唯一测试库，返回路径。
    fn make_temp_db() -> String {
        let path =
            std::env::temp_dir().join(format!("kam_cli_test_{}.sqlite3", uuid::Uuid::new_v4()));
        let conn = Connection::open(&path).expect("open temp db");
        conn.execute(
            "CREATE TABLE auth_kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
            [],
        )
        .expect("create auth_kv");
        path.to_string_lossy().into_owned()
    }

    fn insert_kv(db_path: &str, key: &str, value: &str) {
        let conn = Connection::open(db_path).expect("open db");
        write_kv_value(&conn, key, value).expect("write kv");
    }

    fn key_exists(db_path: &str, key: &str) -> bool {
        let conn = Connection::open(db_path).expect("open db");
        read_kv_value(&conn, key).is_ok()
    }

    #[test]
    fn logout_removes_all_token_keys_but_keeps_device_registration() {
        let db = make_temp_db();
        insert_kv(&db, "kirocli:social:token", "{\"a\":1}");
        insert_kv(&db, "codewhisperer:odic:token", "{\"b\":2}");
        // device-registration 不是 token key，退出登录应当保留它（设备注册信息，重新登录复用）
        insert_kv(&db, "kirocli:social:device-registration", "{\"reg\":true}");

        let removed = logout_cli_account(&db).expect("logout ok");
        assert_eq!(removed, 2, "应删除 2 个 token key");

        assert!(!key_exists(&db, "kirocli:social:token"));
        assert!(!key_exists(&db, "codewhisperer:odic:token"));
        assert!(
            key_exists(&db, "kirocli:social:device-registration"),
            "device-registration 应保留"
        );

        let _ = std::fs::remove_file(&db);
    }

    #[test]
    fn logout_is_idempotent_when_no_token_keys() {
        let db = make_temp_db();
        // 只有 device-registration，没有任何 token key
        insert_kv(&db, "kirocli:social:device-registration", "{\"reg\":true}");

        let removed = logout_cli_account(&db).expect("logout ok");
        assert_eq!(removed, 0, "本来就没登录，删除数量为 0");
        assert!(key_exists(&db, "kirocli:social:device-registration"));

        // 再调一次依旧返回 0，且不报错（幂等）
        let removed_again = logout_cli_account(&db).expect("logout ok again");
        assert_eq!(removed_again, 0);

        let _ = std::fs::remove_file(&db);
    }

    #[test]
    fn logout_errors_when_db_missing() {
        let missing =
            std::env::temp_dir().join(format!("kam_cli_missing_{}.sqlite3", uuid::Uuid::new_v4()));
        let result = logout_cli_account(&missing.to_string_lossy());
        assert!(result.is_err(), "数据库不存在应返回 Err");
    }
}
