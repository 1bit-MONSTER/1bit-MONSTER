/**
 * POST /api/checkout
 * Creates a Stripe Checkout Session for the store cart and returns the
 * hosted checkout URL. Uses only Stripe's REST API — no SDK dependency.
 *
 * Environment variable required (Cloudflare Pages → Settings → Environment variables):
 *   STRIPE_SECRET_KEY = sk_live_...   (or sk_test_... for testing)
 *
 * Request body:
 *   {
 *     "items": [{ "name": "Sticker", "price": 499, "qty": 2 }, ...],
 *     "successUrl": "https://1bit.monster/store/success",
 *     "cancelUrl": "https://1bit.monster/store/"
 *   }
 *
 * Response: { "url": "https://checkout.stripe.com/..." }
 */
export async function onRequest(context) {
  const { request, env } = context;

  if (request.method !== 'POST') {
    return json({ error: 'Method not allowed' }, 405);
  }

  const secretKey = env.STRIPE_SECRET_KEY || '';
  if (!secretKey) {
    return json({ error: 'Store checkout is not configured yet (missing STRIPE_SECRET_KEY).' }, 500);
  }

  let body;
  try {
    body = await request.json();
  } catch {
    return json({ error: 'Invalid JSON body' }, 400);
  }

  const items = (body.items || []).filter(
    (i) => i && typeof i.name === 'string' && Number.isFinite(i.price) && i.price > 0 && Number.isFinite(i.qty) && i.qty > 0
  );
  if (!items.length) {
    return json({ error: 'Cart is empty' }, 400);
  }

  const form = new URLSearchParams();
  form.set('mode', 'payment');
  form.set('success_url', body.successUrl || 'https://1bit.monster/store/success');
  form.set('cancel_url', body.cancelUrl || 'https://1bit.monster/store/');
  items.forEach((item, i) => {
    form.append(`line_items[${i}][quantity]`, String(item.qty));
    form.append(`line_items[${i}][price_data][currency]`, 'usd');
    form.append(`line_items[${i}][price_data][unit_amount]`, String(Math.round(item.price)));
    form.append(`line_items[${i}][price_data][product_data][name]`, String(item.name));
  });

  let resp;
  try {
    resp = await fetch('https://api.stripe.com/v1/checkout/sessions', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${secretKey}`,
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: form.toString(),
    });
  } catch (e) {
    return json({ error: 'Stripe is unreachable: ' + (e.message || e) }, 502);
  }

  const data = await resp.json();
  if (!resp.ok) {
    return json({ error: data.error?.message || `Stripe error (${resp.status})` }, resp.status);
  }
  return json({ url: data.url });
}

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
  });
}
