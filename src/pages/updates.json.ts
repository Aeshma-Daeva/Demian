import { getPublicPosts, postToUpdate } from '../lib/feeds';

export async function GET() {
  const posts = await getPublicPosts();
  return new Response(JSON.stringify({
    generatedAt: new Date().toISOString(),
    updates: posts.map(postToUpdate),
  }, null, 2), {
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
    },
  });
}
