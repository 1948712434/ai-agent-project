import 'dotenv/config';
import { ChatDeepSeek } from "@langchain/deepseek";
import { DynamicTool } from "@langchain/core/tools";
import { createAgent } from "langchain";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import fs from 'fs/promises';

const model = new ChatDeepSeek({
  apiKey: process.env.DEEPSEEK_API_KEY,
  model: "deepseek-chat",
});

// 1. 初始化文本切分器（这就是 RAG 的第一步）
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500, // 每一块最多 500 个字符
  chunkOverlap: 50, // 每块之间重叠 50 个字符，防止一句话被从中间切断
});

// 全局变量，用来保存切分好的文本块
let documentChunks = [];

// 工具 1：加载并切分长文档
const loadAndSplitTool = new DynamicTool({
  name: "loadAndSplitDocument",
  description: "加载并切分本地长文本文件。输入是文件名，例如：long-article.txt",
  func: async (fileName) => {
    try {
      const content = await fs.readFile(fileName, 'utf-8');
      // 调用切分器，把长文本变成数组
      documentChunks = await textSplitter.createDocuments([content]);
      return `文档加载成功，已被切分为 ${documentChunks.length} 个片段。你可以使用 searchDocument 工具来查找具体内容。`;
    } catch (error) {
      return `加载失败: ${error.message}`;
    }
  },
});

// 工具 2：在切分好的文本块里“检索”关键词（这就是 RAG 的第二步：检索）
const searchDocumentTool = new DynamicTool({
  name: "searchDocument",
  description: "在已加载的文档片段中搜索关键词，返回相关的段落。输入应该是你想搜索的关键词或问题。",
  func: async (query) => {
    if (documentChunks.length === 0) {
      return "请先使用 loadAndSplitDocument 工具加载文档！";
    }
    
    // 极简版的“向量检索”：直接找包含关键词的片段
    // （实际生产中这里会用 Embedding 将句子变成向量去算相似度，但原理同理，都是找最相关的）
    const results = documentChunks.filter(chunk => 
      chunk.pageContent.includes(query)
    );

    if (results.length === 0) {
      return `在文档中没有找到与 "${query}" 相关的内容。`;
    }
    
    // 最多返回 2 个最相关的片段，防止 Token 超限
    return results.slice(0, 2).map(r => r.pageContent).join('\n---\n');
  },
});

// 把两个工具交给 Agent
const agent = createAgent({
  model: model,
  tools: [loadAndSplitTool, searchDocumentTool],
});

async function main() {
  // 这里我们要让它先加载，再检索
  const input = "帮我加载 long-article.txt 文件，然后在这份文档里搜一下，关于“前端转行”有什么建议？";
  
  console.log(`🙋 用户提问：${input}\n`);

  const result = await agent.invoke({
    messages: [{ role: "user", content: input }],
  });

  console.log("\n🤖 Agent 分析结果：\n", result.messages.at(-1).content);
}

main();