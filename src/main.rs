// src/main.rs
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

const CHUNK_SIZE: usize = 1000;  // Process 100 records at a time

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

    println!("Completed chunk {}", chunk_num);
    
    // Add a small delay to respect rate limits
    sleep(Duration::from_millis(200)).await;
    
    Ok(embeddings)
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize OpenAI client
    let openai_client = Client::from_env();
    let embedding_model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    // Load and parse customer data
    let file_content = FileLoader::with_glob("data/customer_feedback_satisfaction.csv")?
        .read()
        .into_iter()
        .next()
        .unwrap()?;

    let mut rdr = csv::Reader::from_reader(file_content.as_bytes());
    let customers: Vec<CustomerFeedback> = rdr.deserialize()
        .collect::<Result<Vec<CustomerFeedback>, _>>()?
        .into_iter()
        .map(|mut c| {
            c.profile_summary = String::new();
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

    println!("Generated embeddings for {} records", all_embeddings.len());

    // Create vector store with embeddings
    let vector_store = InMemoryVectorStore::from_documents(all_embeddings);
    let index = vector_store.index(embedding_model);

    // Create the analysis agent
    let agent = openai_client.agent("gpt-4")
        .preamble(r#"
            You are an expert customer insights analyst. Analyze customer profiles and provide:
            1. Key behavioral patterns and trends
            2. Risk factors or concerns
            3. Specific, actionable recommendations
            4. Opportunities for improving customer satisfaction

            Base your analysis on both the specific query and the provided similar customer profiles.
            Be concise but insightful in your analysis.
        "#)
        .build();

    // Build the analysis pipeline
    let chain = pipeline::new()
        .chain(parallel!(
            passthrough::<&str>(),
            lookup::<_, _, CustomerFeedback>(index, 3),
        ))
        .map(|(query, maybe_profiles)| match maybe_profiles {
            Ok(profiles) => format!(
                "Analysis Query: {}\n\nRelevant Customer Profiles for Context:\n{}",
                query,
                profiles.into_iter()
                    .map(|(score, _, profile)| format!(
                        "* Similarity Score: {:.2}\n{}\n",
                        score,
                        profile.profile_summary
                    ))
                    .collect::<String>()
            ),
            Err(err) => {
                eprintln!("Error fetching similar profiles: {}", err);
                query.to_string()
            }
        })
        .prompt(agent);

    // Example queries to test the pipeline
    let example_queries = vec![
        "What patterns do you see in high-income customers with low satisfaction scores?",
        "Analyze the relationship between purchase frequency and loyalty levels.",
        "What characteristics define our most satisfied customers?",
        "Identify potential churn risks based on customer patterns.",
    ];

    for query in example_queries {
        println!("\n=== Query: {} ===\n", query);
        match chain.call(query).await {
            Ok(analysis) => println!("Analysis:\n{}\n", analysis),
            Err(e) => eprintln!("Error analyzing query: {}", e),
        }
    }

    Ok(())
}