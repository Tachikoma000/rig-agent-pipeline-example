# Rig Pipeline Example: Customer Analysis

A simple example demonstrating [Rig's](https://github.com/0xPlaygrounds/rig) agent pipeline. In this example, we build an agent pipeline for analyzing customer feedback and satisfaction data using OpenAI's GPT-4 and text embeddings.

## Features

- Processes large datasets in manageable chunks
- Generates semantic embeddings for customer profiles
- Performs similarity-based searches using vector indices
- Creates analysis pipelines with parallel operations
- Uses GPT-4 to extract insights from customer data

## System Requirements

- Rust (latest stable version)
- OpenAI API key
- 8GB RAM minimum (for processing large datasets)
- 1GB free disk space

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Tachikoma000/rig-agent-pipeline-example
cd rig-pipeline-example
```

2. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
# or create a .env file:
echo "OPENAI_API_KEY=your-key-here" > .env
```

3. Get the dataset:
- Download the synthetic customer feedback dataset (38,444 records) from [Kaggle](https://www.kaggle.com/datasets/jahnavipaliwal/customer-feedback-and-satisfaction)
- Create a `data` directory in the project root if it doesn't exist: `mkdir -p data`
- Place the downloaded `customer_feedback_satisfaction.csv` file in the `data/` directory
- Verify the path: `data/customer_feedback_satisfaction.csv`

4. Build the project:
```bash
cargo build --release
```

## Usage

Run the analysis pipeline:
```bash
cargo run --release
```

The program will:
1. Load customer data from CSV
2. Generate embeddings in batches
3. Create a vector index
4. Run example analysis queries
5. Output insights based on similar customer profiles

## Dataset Schema

The customer feedback dataset includes:

| Field | Type | Description |
|-------|------|-------------|
| CustomerID | String | Unique identifier |
| Age | Integer | Customer age |
| Gender | String | Customer gender |
| Country | String | Country of residence |
| Income | Float | Annual income |
| ProductQuality | Integer | Rating 1-10 |
| ServiceQuality | Integer | Rating 1-10 |
| PurchaseFrequency | Integer | Purchases per year |
| FeedbackScore | String | Customer feedback |
| LoyaltyLevel | String | Customer loyalty |
| SatisfactionScore | Float | Overall satisfaction |

## Project Structure

```
.
├── src/
│   ├── main.rs       # Pipeline implementation
│   └── models.rs     # Data structures
├── data/
│   └── .gitkeep     # Place dataset here
├── Cargo.toml       # Dependencies
└── README.md        # Documentation
```

## Configuration

Adjust these constants in `main.rs`:

- `CHUNK_SIZE`: Number of records processed per batch (default: 1000)

## Example Queries

The pipeline analyzes patterns in:
- High-income customers with low satisfaction
- Purchase frequency vs loyalty correlation
- Service quality across regions
- Churn risk indicators
- Customer satisfaction drivers

## Error Handling

The application:
- Validates dataset presence
- Handles API rate limits
- Manages memory efficiently
- Reports processing errors
- Continues operation after chunk failures

## Dependencies

- `rig-core`: Core Rig framework
- `tokio`: Async runtime
- `serde`: Data serialization
- `csv`: CSV parsing
- `tracing`: Logging and diagnostics

