import readline from "readline";
import fs from "fs";
import { MsgPackReader } from "./msgpack.mjs";
import { Encoder } from "./encoder.mjs";
import { generate } from "./gpt2.mjs";

const variant = "124M";

function main() {
  const encoder = new Encoder(variant);
  const hparams = JSON.parse(
    fs.readFileSync(`./models/${variant}/hparams.json`).toString()
  );

  const msgPackReader = new MsgPackReader(
    `./models/${variant}/${variant}.msgpack`
  );

  const model = msgPackReader.run();

  console.log(
    `GPTv2[${variant}] in JS. Type your prompt and hit enter. Ctrl-C aborts the response.`
  );

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  rl.prompt();

  let running = false;
  let aborted = false;
  rl.on("SIGINT", () => {
    if (!running) {
      process.stdout.write("\n");
      process.exit();
    }
    aborted = true;
  });

  const sleep = (ms) =>
    new Promise((resolve) => setTimeout(() => resolve(), ms));

  rl.on("line", async (prompt) => {
    if (prompt) {
      running = true;

      // encode the input string using the BPE tokenizer
      const inputIds = encoder.encode(prompt);
      const tokensToGenerate = hparams.n_ctx - inputIds.length;
      if (tokensToGenerate < 1) {
        throw new Error("Prompt is too long");
      }

      try {
        // generate output ids
        await generate(
          inputIds,
          model,
          hparams,
          tokensToGenerate,
          async (token) => {
            if (aborted) throw new Error("break");
            // decode the ids back into a string
            process.stdout.write(encoder.decode([token]));
            await sleep(0); // allow event loop to run
          }
        );
      } catch (err) {
        if (err.message !== "break") throw err;
      }

      process.stdout.write("\n");

      aborted = false;
      running = false;
    }
    rl.prompt();
  });
}

main();
