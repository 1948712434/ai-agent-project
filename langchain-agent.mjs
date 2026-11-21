import 'dotenv/config';
import { ChatDeepSeek } from "@langchain/deepseek";
import { DynamicTool } from "@langchain/core/tools";
import { createAgent } from "langchain";
import fs from 'fs/promises';

const model = new ChatDeepSeek({
  apiKey: process.env.DEEPSEEK_API_KEY,
  model: "deepseek-chat",
});

// 工具 1：查看当前目录有哪些文件
const listFilesTool = new DynamicTool({
  name: "listFiles",
  description: "列出当前目录下的所有文件名。当用户想找某个文件但不知道确切名字时，使用此工具。",
  func: async () => {
    try {
      const files = await fs.readdir("./");
      return files.filter(f => !f.startsWith('.') && f !== 'node_modules').join(', '); // 过滤掉隐藏文件和依赖包
    } catch (error) {
      return `读取目录失败: ${error.message}`;
    }
  },
});

// 工具 2：读取指定文件的内容（和之前一样）
const readFileTool = new DynamicTool({
  name: "readLocalFile",
  description: "读取本地文件的内容。输入必须是一个完整的文件名，例如：test-page.html",
  func: async (fileName) => {
    try {
      const content = await fs.readFile(fileName, 'utf-8');
      return content.slice(0, 3000);
    } catch (error) {
      return `读取文件失败: ${error.message}`;
    }
  },
});

// 把两个工具都交给 Agent
const agent = createAgent({
  model: model,
  tools: [listFilesTool, readFileTool], // 注意这里加了 listFilesTool
});

async function main() {
  // 这里把问题变复杂：让模型自己去看目录，然后自己挑文件
  const input = "帮我看一下当前目录下有哪些文件，然后挑出那个叫 test-page 的 HTML 文件，告诉我里面有几个链接？";
  
  console.log(`🙋 用户提问：${input}\n`);

  const result = await agent.invoke({
    messages: [{ role: "user", content: input }],
  });

  console.log("\n🤖 Agent 分析结果：", result.messages.at(-1).content);
}

main();