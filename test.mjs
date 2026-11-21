import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "https://api.deepseek.com",
  apiKey: 'sk-e1baae07f7584c288c96a60ba6401f52',
});

async function main() {
  const completion = await openai.chat.completions.create({
    messages: [{ role: "user", content: "用一句话解释什么是 Agent" }],
    model: "deepseek-flash",
  });
  console.log(completion.choices[0].message.content);
}

main();