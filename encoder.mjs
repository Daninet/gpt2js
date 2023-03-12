import fs from "fs";

const bytesToUnicode = () => {
  const range = (start, end) =>
    [...Array(end - start)].map((_, i) => i + start);

  const bs = [
    ...range("!".charCodeAt(0), "~".charCodeAt(0) + 1),
    ...range("¡".charCodeAt(0), "¬".charCodeAt(0) + 1),
    ...range("®".charCodeAt(0), "ÿ".charCodeAt(0) + 1),
  ];

  const cs = [...bs];

  let n = 0;
  for (let b = 0; b < 2 ** 8; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(2 ** 8 + n);
      n++;
    }
  }

  return Object.fromEntries(bs.map((b, i) => [b, String.fromCharCode(cs[i])]));
};

const byteEncoder = bytesToUnicode();
const byteDecoder = Object.fromEntries(
  Object.entries(byteEncoder).map(([k, v]) => [v, k])
);

const getPairs = (str) => {
  const pairs = new Set();
  let prev = str[0];
  for (let i = 1; i < str.length; i++) {
    pairs.add(prev + " " + str[i]);
    prev = str[i];
  }
  return [...pairs];
};

const pat =
  /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

export class Encoder {
  constructor(variant) {
    this.encoder = JSON.parse(
      fs.readFileSync(`models/${variant}/encoder.json`).toString()
    );

    this.decoder = Object.fromEntries(
      Object.entries(this.encoder).map(([k, v]) => [v, k])
    );

    const bpeData = fs
      .readFileSync(`models/${variant}/vocab.bpe`)
      .toString()
      .split("\n")
      .slice(1, -1);

    this.bpeRanks = Object.fromEntries(bpeData.map((b, i) => [b, i]));
  }

  bpe(token) {
    let pairs = getPairs(token);
    if (!pairs.length) return [token];
    let word = token.split("");

    while (true) {
      const minVal = pairs.reduce(
        (min, curr) => Math.min(this.bpeRanks[curr] ?? Infinity, min),
        Infinity
      );
      if (minVal === Infinity) break;
      const bigram = pairs.find((p) => this.bpeRanks[p] === minVal);
      const [first, second] = bigram.split(" ");
      const newWord = [];
      let i = 0;
      while (i < token.length) {
        let j = word.indexOf(first, i);
        if (j === -1) {
          newWord.push(...word.slice(i));
          break;
        } else {
          newWord.push(...word.slice(i, j));
          i = j;
        }

        if (
          word[i] === first &&
          i < word.length - 1 &&
          word[i + 1] === second
        ) {
          newWord.push(first + second);
          i += 2;
        } else {
          newWord.push(word[i]);
          i += 1;
        }
      }
      word = newWord;

      if (word.length === 1) break;
      pairs = getPairs(word);
    }

    return word;
  }

  encode(str) {
    const bpeTokens = [];

    for (const match of str.matchAll(pat)) {
      let [token] = match;
      const utf8Text = textEncoder.encode(token);
      token = Array.from(utf8Text)
        .map((v) => byteEncoder[v])
        .join("");

      token = this.bpe(token);
      bpeTokens.push(...token.map((t) => this.encoder[t]));
    }
    return bpeTokens;
  }

  decode(tokens) {
    let text = tokens.map((t) => this.decoder[t]).join("");
    text = textDecoder.decode(
      new Uint8Array(text.split("").map((c) => byteDecoder[c]))
    );
    return text;
  }
}
