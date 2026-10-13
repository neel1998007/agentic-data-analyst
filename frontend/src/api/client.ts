/**
 * src/api/client.ts
 * Typed fetch wrapper for the agentic-data-analyst FastAPI backend.
 */

import type { AskResponse } from '../types/api';

// Use a relative URL so the request always goes to the same origin.
// In development (Vite dev server) configure the proxy in vite.config.ts;
// in Docker, Nginx proxies /ask and /health to the FastAPI backend.
const API_BASE_URL = '';

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
      `Network request failed – could not reach /ask. Is the backend running? (${networkError})`
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
