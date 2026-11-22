use serde::{Deserialize, Serialize};
use crate::scheduler::NodeCapabilities;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Message {
    TaskRequest {
        task_id: String,
        prompt: String,
        model_name: String,
        download_url: Option<String>,
        layer_range: Option<(usize, usize)>,
    },
    TaskResponse {
        task_id: String,
        result: Result<String, String>,
    },
    NodeHeartbeat {
        rpc_port: u16,
        capabilities: NodeCapabilities,
    },
}
