use async_openai::{Client, config::OpenAIConfig, types::chat::{ChatCompletionMessageToolCalls::Function, ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage, ChatCompletionRequestToolMessageArgs, ChatCompletionRequestUserMessageArgs, ChatCompletionTools, CreateChatCompletionRequestArgs}};
use clap::Parser;
use std::{collections::HashMap, env, process};

use crate::tools::Tool;

mod tools;

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

    let registry = tools::registry();
    let tools: Vec<ChatCompletionTools> = registry.iter().map(|tool| tool.chat_completion_tools()).collect();
    let by_name: HashMap<String, &Box<dyn Tool>> = registry.iter().map(|tool| (tool.name().to_owned(), tool)).collect();
    
    let mut messages = vec![
        ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
            .content(args.prompt)
            .build()?)
    ];

    let mut request_args = CreateChatCompletionRequestArgs::default();
    request_args
        .model("anthropic/claude-haiku-4.5")
        .tools(tools);

    let mut response = client
        .chat()
        .create(request_args
            .clone()
            .messages(messages.clone())
            .build()?)
        .await?;
    
    
    while let Some(tool_calls) = &response.choices[0].message.tool_calls {
        
        let mut assistant_message_args = ChatCompletionRequestAssistantMessageArgs::default();
        if let Some(content) = &response.choices[0].message.content {
            assistant_message_args.content(ChatCompletionRequestAssistantMessageContent::Text(content.clone()));
        }
        assistant_message_args.tool_calls(tool_calls.clone());
        
        messages.push(ChatCompletionRequestMessage::Assistant(
            assistant_message_args.build()?
        ));
        
        for call in tool_calls.iter().filter_map(|call| {
            let Function(function_call) = call else {
                return None
            };
            Some(function_call)
        }) {
            let content = by_name.get(&call.function.name).unwrap().execute(&call.function.arguments)?;

            messages.push(ChatCompletionRequestMessage::Tool(
                ChatCompletionRequestToolMessageArgs::default()
                    .content(content)
                    .tool_call_id(&call.id)
                    .build()?
            ));
        }
        response = client
            .chat()
            .create(request_args
                .clone()
                .messages(messages.clone())
                .build()?)
            .await?;
    }

    if let Some(content) = &response.choices[0].message.content {
       println!("{}", content);
    }
    
    Ok(())
}
