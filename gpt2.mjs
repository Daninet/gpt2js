import * as tf from "@tensorflow/tfjs-node";

const PIconst = Math.sqrt(2 / Math.PI);

const gelu = (x) => {
  return x
    .pow(3)
    .mul(0.044715)
    .add(x)
    .mul(PIconst)
    .tanh()
    .add(1)
    .mul(x)
    .mul(0.5);
};

const layerNorm = (x, { g, b }, eps = 1e-5) => {
  const { mean, variance } = tf.moments(x, -1);

  // normalize x to have mean=0 and var=1 over last axis
  x = x.sub(mean.expandDims(1)).div(variance.add(eps).sqrt().expandDims(1));

  // scale and offset with gamma/beta params
  return g.mul(x).add(b);
};

const linear = (x, { w, b }) => {
  return x.matMul(w).add(b);
};

const ffn = (x, cFc, cProj) => {
  // project up
  const a = gelu(linear(x, cFc));

  // project back down
  return linear(a, cProj);
};

const attention = (q, k, v, mask) => {
  const op1 = q.matMul(k.transpose());
  const op2 = op1.div(Math.sqrt(q.shape.at(-1)));
  const op3 = op2.add(mask);
  const op4 = op3.softmax();
  return op4.matMul(v);
};

const mha = (x, { c_attn, c_proj }, n_head) => {
  // qkv projection
  x = linear(x, c_attn);

  // split into qkv
  const qkv = x.split(3, -1);

  // split into heads
  const qkvHeads = qkv.map((q) => q.split(n_head, -1));

  // causal mask to hide future inputs from being attended to
  let causalMask = tf.ones([x.shape[0], x.shape[0]], x.dtype);
  causalMask = tf.linalg.bandPart(causalMask, -1, 0);
  causalMask = tf.scalar(1).sub(causalMask).mul(-1e10);

  // perform attention over each head
  const outHeads = qkvHeads[0].map((_, i) =>
    attention(qkvHeads[0][i], qkvHeads[1][i], qkvHeads[2][i], causalMask)
  );

  // merge heads
  x = tf.concat(outHeads, -1);

  // out projection
  return linear(x, c_proj);
};

const transformerBlock = (x, { mlp, attn, ln_1, ln_2 }, n_head) => {
  // multi-head causal self attention
  const mhaRes = mha(layerNorm(x, ln_1), attn, n_head);
  x = x.add(mhaRes);

  // position-wise feed forward network
  const ffnRes = ffn(layerNorm(x, ln_2), mlp.c_fc, mlp.c_proj);
  return x.add(ffnRes);
};

const gpt2 = (inputs, { wte, wpe, blocks, ln_f }, n_head) => {
  // token + positional embeddings

  const wteRows = wte.gather(inputs);
  const wpeRows = wpe.slice([0, 0], [inputs.length]);

  let xArr = wteRows.add(wpeRows);

  // forward pass through n_layer transformer blocks
  for (const block of blocks) {
    xArr = tf.tidy(() => transformerBlock(xArr, block, n_head));
  }

  // projection to vocab
  xArr = layerNorm(xArr, ln_f);
  return xArr.matMul(wte.transpose());
};

export const generate = async (
  tokens,
  model,
  { n_head },
  tokensToGenerate,
  callback
) => {
  const inputTokens = [...tokens];
  // auto-regressive decode loop
  for (let i = 0; i < tokensToGenerate; i++) {
    const token = tf.tidy(() => {
      // model forward pass
      const logits = gpt2(inputTokens, model, n_head);
      // greedy sampling
      const nextId = logits.gather(logits.shape[0] - 1).argMax();
      return nextId.dataSync()[0];
    });

    // append prediction to input
    inputTokens.push(token);

    // export token and allow stopping
    await callback(token);
  }
};
