/**
 * src/types/api.ts
 * TypeScript interfaces that mirror the FastAPI backend's request/response shapes.
 */

export interface AskRequest {
  question: string;
}

export interface AskResponse {
  answer: string;
  tools_called: string[];
}
