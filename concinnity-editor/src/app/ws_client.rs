// src/app/ws_client.rs
//
// WebSocket client for the server command channel.
//
// Connects to the server's GET /v1/ws?account_id=<id> endpoint and receives
// JSON commands, dispatches them through process_command, and sends back a
// JSON ack for each one.
//
// The connection runs on a dedicated background thread with its own
// single-threaded Tokio runtime so it never blocks the synchronous world loop.
// Commands are forwarded over an std::sync::mpsc channel; the caller drains
// it each tick via drain_commands().
//
// Protocol (mirrors src/ws.rs on the server):
//   server → client:
//     {"id":"<uuid>","cmd":"add","type":"Prop","name":"rock","args":{...}}
//     {"id":"<uuid>","cmd":"rm","name":"rock"}
//     {"id":"<uuid>","cmd":"load","assets":[{...}, ...]}
//     {"id":"<uuid>","cmd":"save"}
//     {"id":"<uuid>","cmd":"write_file","path":"shader.metal","content":"..."}
//     {"id":"<uuid>","cmd":"validate_shader","source":"...","name":"shader.metal"}
//     {"id":"<uuid>","cmd":"fetch_assets","names":["cobblestone.png"],"base_url":"http://...","account_id":"grant"}
//   client → server:
//     {"id":"<uuid>","ok":true}
//     {"id":"<uuid>","ok":true,"output":"<compiler diagnostics>"}
//     {"id":"<uuid>","ok":false,"error":"<message>"}

use crate::app::commands::{AppCommand, CommandEffect, process_command};
use serde::{Deserialize, Serialize};

pub type CmdReceiver = std::sync::mpsc::Receiver<PendingCommand>;

// A command received from the server, not yet executed.
// The caller executes it and resolves the ack closure.
// The fields are read by the binary's run loop drain; the FFI lib only names
// the receiver type.
#[allow(dead_code)]
pub struct PendingCommand {
    pub cmd: AppCommand,
    pub ack: Box<dyn FnOnce(Result<String, String>) + Send>,
}

// Raw JSON shape sent by the server.
#[derive(Deserialize)]
struct RawCommand {
    id: String,
    cmd: String,
    // add
    #[serde(rename = "type", default)]
    asset_type: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    args: Option<serde_json::Value>,
    // load
    #[serde(default)]
    assets: Option<Vec<serde_json::Value>>,
    // write_file
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    content: Option<String>,
    // validate_shader
    #[serde(default)]
    source: Option<String>,
    // fetch_assets
    #[serde(default)]
    names: Option<Vec<String>>,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    account_id: Option<String>,
}

#[derive(Serialize)]
struct Ack {
    id: String,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<String>,
}

// Connect to the server's WS endpoint. `url` is the base server URL
// (e.g. "ws://127.0.0.1:8080/v1/ws"); `account_id` is appended as a query
// parameter and is used by the server to identify and authenticate the session.
//
// Returns a receiver that yields PendingCommands as they arrive. The
// background thread exits when the connection closes or the receiver is dropped.
pub fn connect(url: &str, account_id: &str) -> Result<CmdReceiver, String> {
    let url = format!("{url}?account_id={account_id}&client_type=cn");
    let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<PendingCommand>();

    std::thread::Builder::new()
        .name("ws-client".to_string())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to build ws runtime");

            rt.block_on(async move {
                if let Err(e) = run_loop(&url, cmd_tx).await {
                    tracing::error!("ws client exited: {e}");
                }
            });
        })
        .map_err(|e| format!("failed to spawn ws thread: {e}"))?;

    Ok(cmd_rx)
}

async fn run_loop(
    url: &str,
    cmd_tx: std::sync::mpsc::Sender<PendingCommand>,
) -> Result<(), String> {
    use futures::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;

    let request = url
        .into_client_request()
        .map_err(|e| format!("invalid url: {e}"))?;

    let (ws_stream, _) = tokio_tungstenite::connect_async(request)
        .await
        .map_err(|e| format!("connect failed: {e}"))?;

    tracing::info!("ws connected to {url}");

    let (mut sink, mut stream) = ws_stream.split();

    // Outgoing acks are produced inside callbacks; forwarded here to the sink.
    let (ack_tx, mut ack_rx) = tokio::sync::mpsc::channel::<String>(64);

    loop {
        tokio::select! {
            Some(ack) = ack_rx.recv() => {
                if sink.send(Message::Text(ack)).await.is_err() {
                    break;
                }
            }
            msg = stream.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        let raw: RawCommand = match serde_json::from_str(&text) {
                            Ok(r) => r,
                            Err(e) => {
                                tracing::warn!("ws: unrecognised frame ({e}): {text}");
                                continue;
                            }
                        };

                        let id = raw.id.clone();
                        let ack_tx2 = ack_tx.clone();

                        match parse_command(raw) {
                            Ok(cmd) => {
                                let ack_fn = move |result: Result<String, String>| {
                                    let ack = match result {
                                        Ok(output) => Ack {
                                            id,
                                            ok: true,
                                            error: None,
                                            output: if output.is_empty() {
                                                None
                                            } else {
                                                Some(output)
                                            },
                                        },
                                        Err(e) => Ack {
                                            id,
                                            ok: false,
                                            error: Some(e),
                                            output: None,
                                        },
                                    };
                                    let json = serde_json::to_string(&ack)
                                        .unwrap_or_else(|_| r#"{"id":"?","ok":false}"#.to_string());
                                    let _ = ack_tx2.try_send(json);
                                };
                                let pending = PendingCommand {
                                    cmd,
                                    ack: Box::new(ack_fn),
                                };
                                if cmd_tx.send(pending).is_err() {
                                    // Main thread dropped the receiver; time to exit.
                                    break;
                                }
                            }
                            Err(e) => {
                                tracing::warn!("ws: could not parse command '{id}': {e}");
                                let ack = Ack { id, ok: false, error: Some(e), output: None };
                                let json = serde_json::to_string(&ack).unwrap_or_default();
                                let _ = ack_tx.try_send(json);
                            }
                        }
                    }
                    Some(Ok(Message::Ping(b))) => {
                        let _ = sink.send(Message::Pong(b)).await;
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Err(e)) => {
                        tracing::warn!("ws read error: {e}");
                        break;
                    }
                    _ => {}
                }
            }
        }
    }

    tracing::info!("ws disconnected from {url}");
    Ok(())
}

fn parse_command(raw: RawCommand) -> Result<AppCommand, String> {
    match raw.cmd.as_str() {
        "add" => {
            let asset_type = raw.asset_type.ok_or("add requires 'type'")?;
            Ok(AppCommand::Add {
                asset_type,
                name: raw.name,
                args: raw.args,
            })
        }
        "rm" => {
            let name = raw.name.ok_or("rm requires 'name'")?;
            Ok(AppCommand::Rm { name })
        }
        "load" => {
            let assets = raw.assets.ok_or("load requires 'assets'")?;
            Ok(AppCommand::Load { assets })
        }
        "save" => Ok(AppCommand::Save),
        "write_file" => {
            let path = raw.path.ok_or("write_file requires 'path'")?;
            let content = raw.content.ok_or("write_file requires 'content'")?;
            Ok(AppCommand::WriteFile { path, content })
        }
        "validate_shader" => {
            let source = raw.source.ok_or("validate_shader requires 'source'")?;
            let name = raw.name.unwrap_or_else(|| "shader.metal".to_string());
            Ok(AppCommand::ValidateShader { source, name })
        }
        "fetch_assets" => {
            let names = raw.names.ok_or("fetch_assets requires 'names'")?;
            let base_url = raw.base_url.ok_or("fetch_assets requires 'base_url'")?;
            let account_id = raw.account_id.ok_or("fetch_assets requires 'account_id'")?;
            Ok(AppCommand::FetchAssets {
                names,
                base_url,
                account_id,
            })
        }
        "test_world" => {
            let content = raw.content.ok_or("test_world requires 'content'")?;
            Ok(AppCommand::TestWorld { content })
        }
        other => Err(format!("unknown cmd '{other}'")),
    }
}

// Drain all pending commands from the channel, execute them, and send acks.
// Returns true if any command requested a world rebuild (i.e. Save was processed).
// Call this from the world loop before each world_step().
// Driven by the binary's run loop; unreferenced in the FFI lib build.
#[allow(dead_code)]
pub fn drain_commands(rx: &CmdReceiver) -> bool {
    let mut rebuild = false;
    loop {
        match rx.try_recv() {
            Ok(pending) => {
                let result = process_command(pending.cmd);
                if matches!(result, Ok((_, CommandEffect::Rebuild))) {
                    rebuild = true;
                }
                (pending.ack)(result.map(|(out, _)| out));
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => break,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
        }
    }
    rebuild
}
