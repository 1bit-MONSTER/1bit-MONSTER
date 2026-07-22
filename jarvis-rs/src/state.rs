use crate::rag::KnowledgeBase;
use std::sync::Arc;

pub struct AppState {
    pub client: reqwest::Client,
    pub kb: Arc<KnowledgeBase>,
}

pub type SharedState = Arc<AppState>;
