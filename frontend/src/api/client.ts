/**
 * src/api/client.ts
 * Typed fetch wrapper for the agentic-data-analyst FastAPI backend.
 */

import type { AskResponse } from '../types/api';

const API_BASE_URL = 'http://localhost:8000';

/**
 * Send a natural-language question to the backend agent and return
 * the structured response containing the answer and the tools invoked.
 *
 * @throws {Error} if the network request fails or the server returns a non-2xx status.
 */
export async function askAgent(question: string): Promise<AskResponse> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
  } catch (networkError) {
    throw new Error(
      `Network request failed – is the backend running at ${API_BASE_URL}? (${networkError})`
    );
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // ignore JSON parse errors on error responses
    }
    throw new Error(`Agent API returned ${response.status}: ${detail}`);
  }

  return response.json() as Promise<AskResponse>;
}
