import OpenAI from "openai";
import 'dotenv/config'; // 自动读取 .env 文件里的变量
import { ChatDeepSeek } from "@langchain/deepseek";

const openai = new OpenAI({
  apiKey: process.env.DEEPSEEK_API_KEY, 
  model: "deepseek-chat",

});

const model = new ChatDeepSeek({
  // 这里不要写死字符串了，用 process.env 去读
  apiKey: process.env.DEEPSEEK_API_KEY, 
  model: "deepseek-chat",
});

// 1. 定义一个真实的工具函数（这里用模拟数据）
function getWeather(city) {
  if (city.includes("广州")) return "广州今天晴，28度，适合出门。";
  return "抱歉，我不知道这个城市的天气。";
}

async function main() {
  const messages = [{ role: "user", content: "广州今天天气怎么样？" }];

  // 2. 告诉大模型：你有一个叫 getWeather 的工具
  const tools = [
    {
      type: "function",
      function: {
        name: "getWeather",
        description: "获取指定城市的天气信息",
        parameters: {
          type: "object",
          properties: {
            city: { type: "string", description: "城市名称，例如：广州" },
          },
          required: ["city"],
        },
      },
    },
  ];

  console.log("用户提问：", messages[0].content);

  // 第一次请求：看模型是不是想调用工具
  const response = await openai.chat.completions.create({
    model: "deepseek-flash",
    messages: messages,
    tools: tools,
  });

  const responseMessage = response.choices[0].message;

  // 3. 检查模型是不是决定要调用工具
  if (responseMessage.tool_calls) {
    console.log("🧠 模型决定调用工具了！");
    const toolCall = responseMessage.tool_calls[0];
    const functionName = toolCall.function.name;
    const functionArgs = JSON.parse(toolCall.function.arguments);

    console.log(`🔧 模型想要调用：${functionName}，参数是：`, functionArgs);

    // 4. 在我们的代码里真正执行这个函数
    let functionResult = "";
    if (functionName === "getWeather") {
      functionResult = getWeather(functionArgs.city);
    }

    console.log(`💻 本地函数执行结果：${functionResult}`);

    // 5. 把结果塞回对话，让模型生成最终回答
    messages.push(responseMessage); // 把模型要调工具的意图加进去
    messages.push({
      role: "tool",
      tool_call_id: toolCall.id,
      content: functionResult,
    });

    const finalResponse = await openai.chat.completions.create({
      model: "deepseek-flash",
      messages: messages,
    });

    console.log("🤖 最终回答：", finalResponse.choices[0].message.content);
  } else {
    console.log("模型没想调工具，直接回答了：", responseMessage.content);
  }
}

main();