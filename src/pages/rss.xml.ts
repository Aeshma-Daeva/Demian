import { escapeXml, getPublicPosts, postToUpdate, siteUrl } from '../lib/feeds';

export async function GET() {
  const updates = (await getPublicPosts()).map(postToUpdate);
  const items = updates.map(update => `
    <item>
      <title>${escapeXml(update.title)}</title>
      <link>${escapeXml(update.url)}</link>
      <guid>${escapeXml(update.url)}</guid>
      <pubDate>${new Date(`${update.date}T00:00:00Z`).toUTCString()}</pubDate>
      <description>${escapeXml(update.description)}</description>
      ${update.tags.map(tag => `<category>${escapeXml(tag)}</category>`).join('')}
    </item>`).join('');

  const body = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Demian Lab Updates</title>
    <link>${escapeXml(siteUrl())}</link>
    <description>Public progress updates for Demian EEG adaptation.</description>
    <lastBuildDate>${new Date().toUTCString()}</lastBuildDate>
    ${items}
  </channel>
</rss>`;

  return new Response(body, {
    headers: {
      'Content-Type': 'application/rss+xml; charset=utf-8',
    },
  });
}
