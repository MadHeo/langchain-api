import express from "express";
import cors from "cors";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import dotenv from "dotenv";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const model = new ChatOpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
});

const prompt = ChatPromptTemplate.fromTemplate(
  `Answer the user's question. 
  Context: {context}
  Question : {input}`
);

async function loadDocs() {
  const loader = new PDFLoader("./data/pps_rules.pdf");
  const docs = await loader.load();
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 50,
  });

  const splitDocs = await splitter.splitDocuments(docs);
  const embeddings = new OpenAIEmbeddings();
  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );

  return vectorStore.asRetriever({ k: 2 });
}

let retriever;

app.post("/chat", async (req, res) => {
  try {
    console.log("요청받음");
    if (!retriever) retriever = await loadDocs();

    const chain = await createRetrievalChain({
      combineDocsChain: await createStuffDocumentsChain({
        llm: model,
        prompt,
      }),
      retriever,
    });

    const result = await chain.invoke({ input: req.body.data });
    console.log(result);
    res.json(result.answer);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Something went wrong" });
  }
});

app.listen(5000, () => {
  console.log("Server running on port 5000");
});
