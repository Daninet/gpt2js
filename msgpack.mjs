import fs from "fs";

import * as tf from "@tensorflow/tfjs-node";

// Custom reader for the msgpack file format, which does not keep the whole file in RAM
// Large buffers are not readed from the disk until they are not required

export class MsgPackReader {
  offset = 0;

  constructor(path) {
    this.fd = fs.openSync(path, "r");
  }

  skip(bytes) {
    this.offset += bytes;
  }

  read(bytes) {
    const buf = Buffer.allocUnsafe(bytes);
    const bytesRead = fs.readSync(this.fd, buf, 0, bytes, this.offset);
    this.offset += bytesRead;
    return buf.subarray(0, bytesRead);
  }

  run() {
    return this.parse();
  }

  readLength(bytes) {
    const buf = this.read(bytes);
    if (bytes === 1) return buf.readUInt8(0);
    if (bytes === 2) return buf.readUint16BE(0);
    if (bytes === 4) return buf.readUint32BE(0);
    throw new Error("Invalid");
  }

  readArr(n) {
    const entries = [];
    for (let i = 0; i < n; i++) {
      const value = this.parse();
      entries.push(value);
    }
    return entries;
  }

  readTFTensor(shape, buf) {
    const float32Arr = new Float32Array(
      buf.buffer,
      buf.byteOffset,
      buf.length / 4
    );
    return new tf.tensor(float32Arr, shape);
  }

  readObj(n) {
    const normalEntries = [];
    // lazy records are not loaded into the memory
    const lazyEntries = [];
    for (let i = 0; i < n; i++) {
      let key = this.parse();
      if (typeof key === "function") {
        key = key();
      }
      const value = this.parse();
      if (typeof value === "function") {
        lazyEntries.push([key, value]);
      } else {
        normalEntries.push([key, value]);
      }
    }

    const obj = Object.fromEntries(normalEntries);
    if (obj.nd && obj.type === "<f4" && obj.shape) {
      return () =>
        this.readTFTensor(
          obj.shape,
          lazyEntries.find((x) => x[0].toString() === "data")[1]()
        );
    }

    lazyEntries.forEach(([k, v]) => {
      Object.defineProperty(obj, k, {
        get: () => v(),
        enumerable: true,
      });
    });
    return obj;
  }

  // heavy objects are not kept in the memory
  readBuf(n) {
    const length = this.readLength(n);
    if (length < 64) return this.read(length);
    const offset = this.offset;
    this.offset += length;
    return () => {
      const buf = Buffer.allocUnsafe(length);
      const read = fs.readSync(this.fd, buf, 0, length, offset);
      return buf.subarray(0, read);
    };
  }

  parse() {
    const byte = this.read(1)[0];
    if (!(byte >> 7)) return byte;
    if (byte >> 4 === 0b1000) return this.readObj(byte & 0b1111);
    if (byte >> 4 === 0b1001) return this.readArr(byte & 0b1111);
    if (byte >> 5 === 0b101) return this.read(byte & 0b11111).toString();
    if (byte >> 5 === 0b111) return -(byte & 0b11111);

    switch (byte) {
      case 0xc0:
        return null;
      case 0xc1:
        throw new Error("Invalid command");
      case 0xc2:
        return false;
      case 0xc3:
        return true;
      case 0xc4:
        return this.readBuf(1);
      case 0xc5:
        return this.readBuf(2);
      case 0xc6:
        return this.readBuf(4);

      case 0xca:
        return this.read(4).readFloatBE();
      case 0xcb:
        return this.read(4).readDoubleBE();
      case 0xcc:
        return this.read(1).readUInt8();
      case 0xcd:
        return this.read(2).readUint16BE();
      case 0xce:
        return this.read(2).readUint32BE();
      case 0xcf:
        return this.read(2).readBigUInt64BE();
      case 0xd0:
        return this.read(1).readInt8();
      case 0xd1:
        return this.read(2).readInt16BE();
      case 0xd2:
        return this.read(2).readInt32BE();
      case 0xd3:
        return this.read(2).readBigInt64BE();

      case 0xd9:
        return this.read(this.readLength(1)).toString();
      case 0xda:
        return this.read(this.readLength(2)).toString();
      case 0xdb:
        return this.read(this.readLength(4)).toString();
      case 0xdc:
        return this.readArr(this.readLength(2));
      case 0xdd:
        return this.readArr(this.readLength(4));
      case 0xde:
        return this.readObj(this.readLength(2));
      case 0xdf:
        return this.readObj(this.readLength(4));

      default:
        throw new Error("Unsupported msgpack format");
    }
  }
}
