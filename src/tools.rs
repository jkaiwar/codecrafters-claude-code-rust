use std::{error::Error, fs::{create_dir_all, read_to_string, write}, path::Path, process::Command};

use async_openai::{self, types::chat::{ChatCompletionRequestToolMessageContent, ChatCompletionTool, ChatCompletionTools, FunctionObjectArgs}};
use serde_json::{Value, from_str, json};

pub trait Tool {
    fn name(&self) -> &'static str;
    fn chat_completion_tools(&self) -> ChatCompletionTools;
    fn execute(&self, args: &str) -> Result<ChatCompletionRequestToolMessageContent, Box<dyn Error>>;
}

struct Read;
impl Tool for Read {
    fn name(&self) -> &'static str {
        "Read"
    }

    fn chat_completion_tools(&self) -> ChatCompletionTools {
        ChatCompletionTools::Function(ChatCompletionTool {
            function: FunctionObjectArgs::default()
                .name(self.name())
                .description("Read and return the contents of a file")
                .parameters(json!({
                    "type": "object",
                    "required": ["file_path"],
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "the path to the file to read",
                        },
                    },
                }))
                .build()
                .unwrap(),
        })
    }

    fn execute(&self, args: &str) -> Result<ChatCompletionRequestToolMessageContent,Box<dyn Error>>{
        let file_path = from_str::<Value>(args)?
            .get("file_path")
            .and_then(Value::as_str)
            .unwrap()
            .to_owned();
        Ok(ChatCompletionRequestToolMessageContent::Text(read_to_string(file_path)?))
    }
}

struct Write;
impl Tool for Write{
    fn name(&self) -> &'static str {
        "Write"
    }

    fn chat_completion_tools(&self) -> ChatCompletionTools {
        ChatCompletionTools::Function(ChatCompletionTool {
            function: FunctionObjectArgs::default()
                .name(self.name())
                .description("Write content to a file")
                .parameters(json!({
                    "type": "object",
                    "required": ["file_path", "content"],
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "the path of the file to write to",
                        },
                        "content" : {
                            "type": "string",
                            "description": "The content to write to the file",
                        },
                    },
                }))
                .build()
                .unwrap(),
        })
    }

    fn execute(&self, args: &str) -> Result<ChatCompletionRequestToolMessageContent, Box<dyn Error>> {
        let arguments: Value = from_str(args)?;
        
        let file_path = arguments.get("file_path")
            .and_then(Value::as_str)
            .unwrap()
            .to_owned();
        
        let content = arguments.get("content")
            .and_then(Value::as_str)
            .unwrap()
            .to_owned();

        let path = Path::new(&file_path);
        if let Some(parent) = path.parent() {
            create_dir_all(parent)?
        }
        write(&file_path, content)?;

        Ok(ChatCompletionRequestToolMessageContent::Text(
            format!("Successfully wrote content to {}", &file_path
            )))
    }
}

struct Bash;
impl Tool for Bash{
    fn name(&self) -> &'static str {
        "Bash"
    }

    fn chat_completion_tools(&self) -> ChatCompletionTools {
        ChatCompletionTools::Function(ChatCompletionTool {
            function: FunctionObjectArgs::default()
                .name(self.name())
                .description("Execute a shell command")
                .parameters(json!({
                    "type": "object",
                    "required": ["command"],
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute"
                        },
                    },
                }))
                .build()
                .unwrap(),
        })
    }

    fn execute(&self, args: &str) -> Result<ChatCompletionRequestToolMessageContent, Box<dyn Error>> {
        let value: Value = from_str(args)?;
        let command = value
            .get("command")
            .and_then(Value::as_str)
            .ok_or("command missing")
            .to_owned()?;
        
        let output = Command::new("bash")
            .arg("-c")
            .arg(command)
            .output()?;

        let message = match output.status.code() {
            Some(code) => if code == 0 {
                format!("program exited successfully: {}", String::from_utf8(output.stdout)?)
            } else {
                format!("program exited with code {}: {}", code, String::from_utf8(output.stderr)?)
            },
            None => format!("program was interrupted"),
        };

        Ok(ChatCompletionRequestToolMessageContent::Text(message))
    }
}

pub fn registry() -> Vec<Box<dyn Tool>> {
    vec![Box::new(Read), Box::new(Write), Box::new(Bash)]
}
