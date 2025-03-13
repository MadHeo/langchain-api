import express from "express";
import cors from "cors";
import { chatApi } from "./api/chat.js";

const app = express();
app.use(cors());
app.use(express.json());

app.post("/chat", chatApi);

app.listen(5000, () => {
  console.log("Server running on port 5000");
});
