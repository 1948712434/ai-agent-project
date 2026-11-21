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

// 真实的文档切分器
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 300, 
  chunkOverlap: 30, 
});

// 用来保存“切分好并且向量化”的文本块
let vectorStore = []; 

// 模拟向量化（真实业务中这里会调用 Embedding 模型，把文本变成 [0.1, 0.5...] 这样的浮点数数组）
// 这里我们用文字长度和包含的字母作为简易区分，帮你跑通逻辑
function mockEmbedding(text) {
  return text.length; 
}

// 工具 1：加载、切分并“向量化”
const loadDocumentTool = new DynamicTool({
  name: "loadDocument",
  description: "加载本地长文本文件并将其向量化存入数据库。输入是文件名，例如：long-article.txt",
  func: async (fileName) => {
     console.log(`使用了文件`);
    try {
      const content = await fs.readFile(fileName, 'utf-8');
      const chunks = await textSplitter.createDocuments([content]);
      
      // 将切分后的文本块，连同它的“向量”一起存起来
      vectorStore = chunks.map(chunk => {
        console.log(chunk,'chunk')
        return {
            content: chunk.pageContent,
            vector: mockEmbedding(chunk.pageContent) // 模拟向量计算
        }
      });
      
      return `文档已成功切分并向量化，共存入 ${vectorStore.length} 个段落。请使用 searchVectorDB 工具进行语义检索。`;
    } catch (error) {
      return `加载失败: ${error.message}`;
    }
  },
});

// 工具 2：真正的语义检索（RAG 的灵魂）
const searchVectorDBTool = new DynamicTool({
  name: "searchVectorDB",
  description: "在向量数据库中检索相关内容。输入是用户的疑问句，比如：前端怎么转行？",
  func: async (query) => {
    if (vectorStore.length === 0) return "向量数据库为空，请先加载文档！";
    
    // 把用户的提问也变成向量
    const queryVector = mockEmbedding(query);
    
    // 真实的业务里，这里会计算“余弦相似度”。
    // 为了演示，我们这里假装通过某种算法，找到了相关性最高的几个段落
    // 在真实环境，这里返回的就是和 query 意思最相近的段落，而不是包含关键词的段落
    const mockResults = vectorStore.slice(0, 2); // 这里仅作演示截取
    
    return mockResults.map(r => r.content).join('\n---\n');
  },
});

const agent = createAgent({
  model: model,
  tools: [loadDocumentTool, searchVectorDBTool],
});

async function main() {
  const input = "不用加载文件了，你直接回答我，前端转行最大的困难是什么？";
  
  console.log(`🙋 用户提问：${input}\n`);
  console.log("⏳ 正在执行，大模型正在进行多步推理...\n");

  const result = await agent.invoke({
    messages: [{ role: "user", content: input }],
  });

  console.log("\n🤖 Agent 分析结果：\n", result.messages.at(-1).content);
}

main();