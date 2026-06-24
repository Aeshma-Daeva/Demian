import { getCollection } from 'astro:content';

export function siteUrl(path = '') {
  const site = import.meta.env.SITE.replace(/\/$/, '');
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  const cleanPath = path.replace(/^\//, '');
  return `${site}${base}/${cleanPath}`.replace(/\/$/, '');
}

export async function getPublicPosts() {
  const posts = await getCollection('posts', ({ data }) => data.status !== 'draft');
  return posts.sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}

export function postToUpdate(post: Awaited<ReturnType<typeof getPublicPosts>>[number]) {
  return {
    title: post.data.title,
    date: post.data.date.toISOString().slice(0, 10),
    description: post.data.description ?? '',
    tags: post.data.tags ?? [],
    url: siteUrl(`log/${post.slug}`),
    artifacts: post.data.artifacts ?? [],
    metrics: post.data.metrics ?? [],
  };
}

export function escapeXml(value: string) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;');
}
