# Rig Agent Pipeline Example

This project demonstrates how to build an Agent pipeline for analyzing customer feedback and satisfaction data. It showcases how to:

1. Load customer data from a CSV file.
2. Generate embeddings for similarity-based retrieval.
3. Store embeddings in an in-memory vector index.
4. Use an LLM (GPT-4) agent to analyze user queries with relevant customer profiles.
5. Return actionable insights based on the analysis.

## Features

- **Flexible Pipelines**: Build modular and composable AI workflows.
- **Embedding Integration**: Use OpenAI's embedding models to represent data semantically.
- **Vector Indexing**: Perform efficient similarity-based lookups.
- **LLM Agents**: Integrate GPT-4 for detailed analysis and insights.

## Project Structure

```
rig-agent-pipeline-example/
├── .gitignore
├── Cargo.toml
├── data/
│   └── customer_feedback_satisfaction.csv
├── src/
│   ├── main.rs
│   └── models.rs
```

### Key Files

- **`models.rs`**: Defines the `CustomerFeedback` struct and implements methods for generating summaries for embedding.
- **`main.rs`**: The main pipeline logic, including data loading, embedding generation, pipeline construction, and example queries.

## Dataset

The project uses a **synthetic dataset** sourced from [Kaggle](https://www.kaggle.com/datasets/jahnavipaliwal/customer-feedback-and-satisfaction), which contains 38,444 customer records. Each record captures demographic and behavioral features:

- `CustomerID`: Unique identifier.
- `Age`, `Gender`, `Country`, `Income`: Demographic data.
- `ProductQuality`, `ServiceQuality`: Ratings (1-10).
- `PurchaseFrequency`: Number of purchases in the last year.
- `FeedbackScore`, `LoyaltyLevel`, `SatisfactionScore`: Feedback and satisfaction metrics.

This dataset is ideal for exploring customer satisfaction metrics and predictive modeling.

## Getting Started

### Prerequisites

1. Install Rust and Cargo ([Get Started with Rust](https://www.rust-lang.org/learn/get-started)).
2. Set up an OpenAI API key and export it as an environment variable:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```
3. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd rig-agent-pipeline-example
   ```
4. Dataset used in this example can be found here: https://www.kaggle.com/datasets/jahnavipaliwal/customer-feedback-and-satisfaction

### Installation

1. Fetch dependencies:
   ```bash
   cargo build
   ```

2. Prepare the dataset:
   - Place the `customer_feedback_satisfaction.csv` file in the `data/` directory.

### Running the Project

Execute the following command to run the pipeline:
```bash
cargo run
```

## How It Works

### 1. Define the Data Model (`models.rs`)

The `CustomerFeedback` struct represents a customer profile. The `profile_summary` field is marked with `#[embed]` for embedding generation.

```rust
#[derive(Debug, Deserialize, Serialize, Clone, Embed, PartialEq)]
pub struct CustomerFeedback {
    // Fields like customer_id, age, income, etc.
    #[embed]
    #[serde(skip)]
    pub profile_summary: String,
}

impl CustomerFeedback {
    pub fn generate_summary(&mut self) {
        self.profile_summary = format!(
            "Customer Profile: {} year old {} from {} with income ${:.2}. ...",
            self.age, self.gender, self.country, self.income
        );
    }
}
```

### 2. Main Pipeline (`main.rs`)

#### Data Loading

Customer data is loaded from the CSV file and parsed into `CustomerFeedback` objects:

```rust
let file_content = FileLoader::with_glob("data/customer_feedback_satisfaction.csv")?.read().into_iter().next().unwrap()?;
let mut rdr = csv::Reader::from_reader(file_content.as_bytes());
let customers: Vec<CustomerFeedback> = rdr.deserialize().collect::<Result<Vec<_>, _>>()?;
```

#### Embedding Generation

Data is split into chunks, and embeddings are generated using OpenAI's embedding model:

```rust
const CHUNK_SIZE: usize = 1000;
let chunks: Vec<Vec<CustomerFeedback>> = customers.chunks(CHUNK_SIZE).map(|c| c.to_vec()).collect();

for chunk in chunks {
    let embeddings = EmbeddingsBuilder::new(embedding_model.clone()).documents(chunk)?.build().await?;
}
```

#### Vector Indexing

Embeddings are stored in an in-memory vector index for fast similarity lookups:

```rust
let vector_store = InMemoryVectorStore::from_documents(all_embeddings);
let index = vector_store.index(embedding_model);
```

#### Pipeline Construction

The pipeline merges user queries with relevant customer profiles and sends the data to GPT-4 for analysis:

```rust
let chain = pipeline::new()
    .chain(parallel!(
        passthrough::<&str>(),
        lookup::<_, _, CustomerFeedback>(index, 3),
    ))
    .map(|(query, profiles)| format!("Query: {}\n\nProfiles: {}", query, profiles))
    .prompt(agent);
```

#### Example Queries

Test the pipeline with predefined queries:

```rust
let queries = vec![
    "What patterns do you see in high-income customers?",
    "Identify potential churn risks.",
];

for query in queries {
    let result = chain.call(query).await?;
    println!("{}", result);
}
```

## Example Output

### Query

> What patterns do you see in high-income customers with low satisfaction scores?

### Response

```
Key Patterns:
- High-income customers demand personalized experiences.
- Sensitivity to service quality and responsiveness.

Risks:
- Negative reviews may impact brand reputation.

Recommendations:
- Tailored loyalty programs for high-income segments.
- Improve customer service for faster issue resolution.
```

## Customization

- Adjust chunk size by modifying `CHUNK_SIZE` in `main.rs`.
- Update the agent's preamble to suit your use case.
- Replace the in-memory vector store with a production-ready database (e.g., Qdrant, MongoDB).

## Dependencies

- [Rig](https://github.com/0xPlaygrounds/rig) (Core library)
- OpenAI API
- Tokio (Async runtime)
- Serde (Serialization/deserialization)
- CSV (CSV parsing)

