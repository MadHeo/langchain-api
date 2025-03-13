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

const llm = new ChatOpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
  temperature: 0,
});

const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-small" }); // 한 번만 생성

const prompt = ChatPromptTemplate.fromTemplate(`
    You are a helpful assistant providing detailed responses.

  Please format your response in multiple paragraphs to improve readability.
  
  Context:
  {context}

  Question:
  {input}
  
  Response:
`);

const loadDocs = async () => {
  const loader = new PDFLoader("./data/company_rules.pdf");
  const docs = await loader.load(); // PDF 로드

  // 로드된 PDF를 스플릿
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const allSplits = await splitter.splitDocuments(docs);

  const vectorStore = await MemoryVectorStore.fromDocuments(
    allSplits,
    embeddings
  );

  return vectorStore.asRetriever({ k: 2 });
};

let retriever;

export const chatApi = async (req, res) => {
  try {
    if (!retriever) retriever = await loadDocs();
    const { question } = req.body;

    // 문서 결합 체인 생성
    const combineDocsChain = await createStuffDocumentsChain({
      llm,
      prompt,
    });

    // 검색 체인 생성
    const chain = await createRetrievalChain({
      combineDocsChain,
      retriever,
    });

    const result = await chain.invoke({ input: question });

    res.json(result);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Something went wrong" });
  }
};
