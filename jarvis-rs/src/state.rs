use crate::rag::KnowledgeBase;
use crate::voice::VoiceEngine;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct AppState {
    pub client: reqwest::Client,
    pub kb: Arc<KnowledgeBase>,
    pub voice: Mutex<VoiceEngine>,
}

pub type SharedState = Arc<AppState>;
