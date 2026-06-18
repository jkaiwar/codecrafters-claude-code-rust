use async_openai::{Client, config::OpenAIConfig, types::chat::{ChatCompletionMessageToolCalls::Function, CreateChatCompletionResponse}};
use clap::Parser;
use serde_json::{Value, from_str, from_value, json};
use std::{env, fs::File, io, process};

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(short = 'p', long)]
    prompt: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let base_url = env::var("OPENROUTER_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());

    let api_key = env::var("OPENROUTER_API_KEY").unwrap_or_else(|_| {
        eprintln!("OPENROUTER_API_KEY is not set");
        process::exit(1);
    });

    let config = OpenAIConfig::new()
        .with_api_base(base_url)
        .with_api_key(api_key);

    let client = Client::with_config(config);

    let response: Value = client
        .chat()
        .create_byot(json!({
            "messages": [
                {
                    "role": "user",
                    "content": args.prompt
                }
            ],
            "model": "anthropic/claude-haiku-4.5",
            "tools": [{
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read and return the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to read",
                            },
                        },
                        "required": ["file_path"],
                    },
                },
            },]
        }))
        .await?;

    let message = &from_value::<CreateChatCompletionResponse>(response)?.choices[0].message;

    if let Some(content) = &message.content {
        println!("{}", content);
    }

    if let Some(tool_calls) = &message
        .tool_calls {
            let Function(call) = &tool_calls[0] else {
                panic!();
            };
            assert_eq!(call.function.name, "Read");
            let file_path = from_str::<Value>(&call.function.arguments)?
                .get("file_path")
                .and_then(Value::as_str)
                .unwrap()
                .to_owned();
            
            let mut file = File::open(file_path)?;
            let mut out = io::stdout();
            io::copy(&mut file, &mut out)?;
        }
    
    Ok(())
}
