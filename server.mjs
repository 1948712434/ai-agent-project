import "dotenv/config";
import express from "express";
import cors from "cors";
import { ChatDeepSeek } from "@langchain/deepseek";
import { DynamicTool } from "@langchain/core/tools";
import { createAgent } from "langchain";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import fs from "fs/promises";

const app = express();
app.use(cors());
app.use(express.json()); // 允许接收 JSON 格式的请求体

const model = new ChatDeepSeek({
  apiKey: process.env.DEEPSEEK_API_KEY,
  model: "deepseek-chat",
});

const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.SILICON_API_KEY,
  configuration: { baseURL: "https://api.siliconflow.cn/v1" },
  modelName: "BAAI/bge-m3",
});

let vectorStore = null;

// 工具 1：加载文档
const loadDocumentTool = new DynamicTool({
  name: "loadDocument",
  description:
    "加载本地长文本文件并将其存入向量数据库。输入是文件名，例如：long-article.txt",
  func: async (fileName) => {
    try {
      const content = await fs.readFile(fileName, "utf-8");
      const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 300,
        chunkOverlap: 30,
      });
      const chunks = await textSplitter.createDocuments([content]);
      vectorStore = await MemoryVectorStore.fromDocuments(chunks, embeddings);
      return `文档已成功向量化！请使用 searchVectorDB 工具查询。`;
    } catch (error) {
      return `加载失败: ${error.message}`;
    }
  },
});

// 工具 2：检索文档
const searchVectorDBTool = new DynamicTool({
  name: "searchVectorDB",
  description: "在向量数据库中检索相关内容。",
  func: async (query) => {
    if (!vectorStore) return "数据库为空，请先加载文档！";
    const results = await vectorStore.similaritySearch(query, 2);
    return results.map((r) => r.pageContent).join("\n---\n");
  },
});

const agent = createAgent({
  model: model,
  tools: [loadDocumentTool, searchVectorDBTool],
});
// ⭐ 接收前端请求的核心 API 接口
app.post("/api/chat", async (req, res) => {
  const userMessage = req.body.message;

  try {
    // 强制让 Agent 每次先加载文档，再回答问题
    const input = `请先加载 long-article.txt 文件。然后回答这个问题：${userMessage}`;

    const result = await agent.invoke({
      messages: [{ role: "user", content: input }],
    });

    // 把结果返回给前端
    res.json({ reply: result.messages.at(-1).content });
  } catch (error) {
    console.error("Agent 运行出错:", error);
    res.status(500).json({ error: error.message });
  }
});
app.use(express.static("."));
// 启动服务器
const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`🚀 服务器运行在端口 ${port}`);
});
