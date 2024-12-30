use rig::Embed;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

#[allow(dead_code)]
#[derive(Debug, Deserialize, Serialize, Clone, Embed, PartialEq)]
pub struct CustomerFeedback {
    #[serde(rename = "CustomerID")]
    pub customer_id: String,
    #[serde(rename = "Age")]
    pub age: i32,
    #[serde(rename = "Gender")]
    pub gender: String,
    #[serde(rename = "Country")]
    pub country: String,
    #[serde(rename = "Income")]
    pub income: f64,
    #[serde(rename = "ProductQuality")]
    pub product_quality: i32,
    #[serde(rename = "ServiceQuality")]
    pub service_quality: i32,
    #[serde(rename = "PurchaseFrequency")]
    pub purchase_frequency: i32,
    #[serde(rename = "FeedbackScore")]
    pub feedback_score: String,
    #[serde(rename = "LoyaltyLevel")]
    pub loyalty_level: String,
    #[serde(rename = "SatisfactionScore")]
    pub satisfaction_score: f64,
    // Field that will be used for embeddings
    #[embed]
    #[serde(skip)]
    pub profile_summary: String,
}

// Implement Eq manually, using only the customer_id for equality comparison
impl Eq for CustomerFeedback {}

impl PartialOrd for CustomerFeedback {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CustomerFeedback {
    fn cmp(&self, other: &Self) -> Ordering {
        self.customer_id.cmp(&other.customer_id)
    }
}

impl CustomerFeedback {
    // Generate a text summary for embedding
    pub fn generate_summary(&mut self) {
        self.profile_summary = format!(
            "Customer Profile: {} year old {} from {} with income ${:.2}. \
             Product Quality Rating: {}/10, Service Quality: {}/10. \
             Purchases {} times per year. Feedback Score: {}. \
             Loyalty Level: {}. Satisfaction Score: {:.1}%",
            self.age, self.gender, self.country, self.income,
            self.product_quality, self.service_quality,
            self.purchase_frequency, self.feedback_score,
            self.loyalty_level, self.satisfaction_score
        );
    }
}
