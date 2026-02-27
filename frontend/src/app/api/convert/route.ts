const API_BASE_URL =
  process.env.WORDTOLATEX_API_BASE_URL || "http://localhost:8000";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const formData = await request.formData();
  const response = await fetch(`${API_BASE_URL}/v1/convert`, {
    method: "POST",
    body: formData,
  });

  const headers = new Headers();
  const contentType = response.headers.get("content-type");
  if (contentType) {
    headers.set("content-type", contentType);
  }
  const disposition = response.headers.get("content-disposition");
  if (disposition) {
    headers.set("content-disposition", disposition);
  }

  if (response.body) {
    return new Response(response.body, {
      status: response.status,
      headers,
    });
  }

  const fallbackBody = await response.arrayBuffer();
  return new Response(fallbackBody, {
    status: response.status,
    headers,
  });
}
