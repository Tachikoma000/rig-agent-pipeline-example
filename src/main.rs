mod models;
use models::CustomerFeedback;
use rig::{
    embeddings::{EmbeddingsBuilder, Embedding},
    parallel,
    pipeline::{self, agent_ops::lookup, passthrough, Op},
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::in_memory_store::InMemoryVectorStore,
    loaders::FileLoader,
    OneOrMany,
};
use std::time::Duration;
use tokio::time::sleep;

const CHUNK_SIZE: usize = 1000;  // Process 1000 records at a time

async fn process_chunk(
    chunk: Vec<CustomerFeedback>,
    embedding_model: &rig::providers::openai::EmbeddingModel,
    chunk_num: usize,
) -> Result<Vec<(CustomerFeedback, OneOrMany<Embedding>)>, anyhow::Error> {
    println!("Processing chunk {} ({} records)...", chunk_num, chunk.len());
    
    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .documents(chunk)?
        .build()
        .await?;

    println!("Completed chunk {} with {} embeddings", chunk_num, embeddings.len());
    
    // Add a small delay to respect rate limits
    sleep(Duration::from_millis(200)).await;
    
    Ok(embeddings)
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Setup logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    // Check for data file
    let data_path = "data/customer_feedback_satisfaction.csv";
    if !std::path::Path::new(data_path).exists() {
        return Err(anyhow::anyhow!("Data file not found: {}", data_path));
    }

    // Initialize OpenAI client
    let openai_client = Client::from_env();
    let embedding_model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    // Load and parse customer data
    let file_content = FileLoader::with_glob(data_path)?
        .read()
        .into_iter()
        .next()
        .unwrap()?;

    let mut rdr = csv::Reader::from_reader(file_content.as_bytes());
    let customers: Vec<CustomerFeedback> = rdr.deserialize()
        .collect::<Result<Vec<CustomerFeedback>, _>>()?
        .into_iter()
        .map(|mut c| {
            c.generate_summary();
            c
        })
        .collect();

    println!("Loaded {} customer records", customers.len());
    
    // Process in chunks
    let chunks: Vec<Vec<CustomerFeedback>> = customers
        .chunks(CHUNK_SIZE)
        .map(|chunk| chunk.to_vec())
        .collect();

    println!("Split into {} chunks of size {}", chunks.len(), CHUNK_SIZE);

    // Process all chunks
    let mut all_embeddings = Vec::new();
    for (chunk_num, chunk) in chunks.into_iter().enumerate() {
        match process_chunk(chunk, &embedding_model, chunk_num + 1).await {
            Ok(embeddings) => all_embeddings.extend(embeddings),
            Err(e) => {
                eprintln!("Error processing chunk {}: {}", chunk_num + 1, e);
                continue;
            }
        }
    }

    println!("Generated {} embeddings with dimension {}", 
        all_embeddings.len(),
        all_embeddings.first().map(|(_, e)| e.first().vec.len()).unwrap_or(0)
    );

    // Create vector store with embeddings
    let vector_store = InMemoryVectorStore::from_documents(all_embeddings);
    let index = vector_store.index(embedding_model);

    // Create the analysis agent
    let agent = openai_client.agent("gpt-4")
        .preamble(r#"
            You are an expert customer insights analyst. You will be provided with:
            1. A specific analysis query
            2. Several relevant customer profiles with detailed metrics including:
               - Demographics (age, gender, country)
               - Income level
               - Product and service quality ratings
               - Purchase frequency
               - Feedback scores
               - Loyalty level
               - Satisfaction scores

            Analyze the provided profiles in relation to the query and provide:
            1. Key behavioral patterns and trends from the specific profiles shown
            2. Risk factors or concerns based on the actual data
            3. Specific, actionable recommendations
            4. Opportunities for improving customer satisfaction

            Always reference specific data points from the provided profiles to support your analysis.
            Be concise but insightful.
        "#)
        .build();

    // Build the analysis pipeline
    let chain = pipeline::new()
        .chain(parallel!(
            passthrough::<&str>(),
            lookup::<_, _, CustomerFeedback>(index, 5),
        ))
        .map(|(query, maybe_profiles)| match maybe_profiles {
            Ok(profiles) => {
                if profiles.is_empty() {
                    format!("Analysis Query: {}\n\nWarning: No relevant customer profiles found.", query)
                } else {
                    format!(
                        "Analysis Query: {}\n\nRelevant Customer Profiles ({} found):\n{}",
                        query,
                        profiles.len(),
                        profiles.into_iter()
                            .enumerate()
                            .map(|(i, (score, _, profile))| format!(
                                "Profile {}:\n* Similarity Score: {:.3}\n* Customer ID: {}\n* Demographics: {} year old {} from {}\n* Income: ${:.2}\n* Satisfaction: {:.1}%\n* Loyalty Level: {}\n* Purchase Frequency: {} purchases/year\n* Product Quality: {}/10\n* Service Quality: {}/10\n* Feedback Score: {}\n",
                                i + 1,
                                score,
                                profile.customer_id,
                                profile.age,
                                profile.gender,
                                profile.country,
                                profile.income,
                                profile.satisfaction_score,
                                profile.loyalty_level,
                                profile.purchase_frequency,
                                profile.product_quality,
                                profile.service_quality,
                                profile.feedback_score
                            ))
                            .collect::<String>()
                    )
                }
            },
            Err(err) => {
                eprintln!("Error retrieving similar profiles: {}", err);
                format!("Analysis Query: {}\n\nError: Failed to retrieve relevant customer profiles.", query)
            }
        })
        .prompt(agent);

    // Example queries to test the pipeline
    let example_queries = vec![
        "What patterns do you see in high-income customers with low satisfaction scores?",
        "Analyze the relationship between purchase frequency and loyalty levels.",
        "What characteristics define our most satisfied customers?",
        "Identify potential churn risks based on customer patterns.",
        "Find patterns in service quality ratings across different countries.",
    ];

    for query in example_queries {
        println!("\n=== Query: {} ===\n", query);
        match chain.call(query).await {
            Ok(analysis) => println!("Analysis:\n{}\n", analysis),
            Err(e) => eprintln!("Error analyzing query: {}", e),
        }
        // Add a small delay between queries
        sleep(Duration::from_secs(2)).await;
    }

    Ok(())
}
