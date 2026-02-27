const API_BASE_URL =
  process.env.WORDTOLATEX_API_BASE_URL || "http://localhost:8000";

export const dynamic = "force-dynamic";

export async function GET() {
  const response = await fetch(`${API_BASE_URL}/stats`, { cache: "no-store" });
  const headers = new Headers();
  const contentType = response.headers.get("content-type");
  if (contentType) {
    headers.set("content-type", contentType);
  } else {
    headers.set("content-type", "application/json");
  }

  if (response.body) {
    return new Response(response.body, {
      status: response.status,
      headers,
    });
  }

  const payload = await response.text();
  return new Response(payload, {
    status: response.status,
    headers,
  });
}
