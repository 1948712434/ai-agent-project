import "dotenv/config";
import { ChatDeepSeek } from "@langchain/deepseek";
import { DynamicTool } from "@langchain/core/tools";
import { createAgent } from "langchain";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import fs from "fs/promises";

// 1. 配置 DeepSeek 聊天模型
const model = new ChatDeepSeek({
  apiKey: process.env.DEEPSEEK_API_KEY,
  model: "deepseek-chat",
});

// 2. 配置真正的向量化模型（借用硅基流动的免费 BGE 模型，用 OpenAI 协议兼容）
const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.SILICON_API_KEY,
  configuration: {
    baseURL: "https://api.siliconflow.cn/v1", // 指向硅基流动的服务器
  },
  modelName: "BAAI/bge-m3", // 这是目前最好用的开源向量模型之一
});

// 全局变量：真实的向量数据库
let vectorStore = null;

// 工具 1：加载、切分并存入真实向量库
const loadDocumentTool = new DynamicTool({
  name: "loadDocument",
  description:
    "加载本地长文本文件并将其存入向量数据库。输入是文件名，例如：long-article.txt",
  func: async (fileName) => {
    try {
      const content = await fs.readFile(fileName, "utf-8");

      // 切分文本
      const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 300,
        chunkOverlap: 30,
      });
      const chunks = await textSplitter.createDocuments([content]);

      // 真正调用 Embedding API，把文字变成数字向量，并存进内存库
      vectorStore = await MemoryVectorStore.fromDocuments(chunks, embeddings);

      return `文档已成功向量化！共存入 ${chunks.length} 个片段。请使用 searchVectorDB 工具查询。`;
    } catch (error) {
      return `加载失败: ${error.message}`;
    }
  },
});

// 工具 2：真正的语义检索
const searchVectorDBTool = new DynamicTool({
  name: "searchVectorDB",
  description:
    "在向量数据库中检索相关内容。输入是用户的疑问句，比如：前端怎么转行？",
  func: async (query) => {
    if (!vectorStore) return "数据库为空，请先加载文档！";

    // 真实计算向量相似度，找出最相关的 2 个段落
    const results = await vectorStore.similaritySearch(query, 2);

    return results.map((r) => r.pageContent).join("\n---\n");
  },
});

const agent = createAgent({
  model: model,
  tools: [loadDocumentTool, searchVectorDBTool],
});

async function main() {
  // 测试一下语义检索的威力（注意提问方式和文档里的词汇不一样，看看它能不能找出来）
  const input = "我是一名本科前端，想转行做 AI 开发，有什么建议？";

  console.log(`🙋 用户提问：${input}\n`);
  console.log("⏳ 正在进行真实的向量计算，请稍候...\n");

  const result = await agent.invoke({
    messages: [{ role: "user", content: input }],
  });

  console.log("\n🤖 Agent 分析结果：\n", result.messages.at(-1).content);
}

main();
