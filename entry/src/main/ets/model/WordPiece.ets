export const SEQUENCE_LENGTH: number = 64;

export class TokenizedInput {
  ids: Int32Array = new Int32Array(SEQUENCE_LENGTH);
  mask: Int32Array = new Int32Array(SEQUENCE_LENGTH);
  tooLong: boolean = false;
}

export class WordPiece {
  private readonly vocabulary: Map<string, number> = new Map<string, number>();
  private readonly pad: number;
  private readonly unknown: number;
  private readonly start: number;
  private readonly end: number;

  constructor(vocab: string) {
    const lines: string[] = vocab.replace(/\r/g, '').split('\n');
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].length > 0) {
        this.vocabulary.set(lines[i], i);
      }
    }
    this.pad = this.requiredId('[PAD]');
    this.unknown = this.requiredId('[UNK]');
    this.start = this.requiredId('[CLS]');
    this.end = this.requiredId('[SEP]');
  }

  encode(text: string): TokenizedInput {
    const normalized: string = text.toLowerCase().normalize('NFD').replace(/\p{Mn}/gu, '');
    const tokenIds: number[] = [];
    let word: string = '';
    for (const char of normalized) {
      const code: number = char.codePointAt(0) ?? 0;
      if (char === ' ' || char === '\t' || char === '\n' || char === '\r' || /\p{Zs}/u.test(char)) {
        this.appendWord(word, tokenIds);
        word = '';
      } else if (code !== 0 && code !== 0xFFFD && !/\p{C}/u.test(char)) {
        if (this.isChinese(code) || this.isPunctuation(code, char)) {
          this.appendWord(word, tokenIds);
          word = '';
          tokenIds.push(this.vocabulary.get(char) ?? this.unknown);
        } else {
          word += char;
        }
      }
    }
    this.appendWord(word, tokenIds);
    const input: TokenizedInput = new TokenizedInput();
    input.ids.fill(this.pad);
    input.tooLong = tokenIds.length > SEQUENCE_LENGTH - 2;
    const usableCount: number = Math.min(tokenIds.length, SEQUENCE_LENGTH - 2);
    input.ids[0] = this.start;
    for (let i = 0; i < usableCount; i++) {
      input.ids[i + 1] = tokenIds[i];
    }
    input.ids[usableCount + 1] = this.end;
    input.mask.fill(1, 0, usableCount + 2);
    return input;
  }

  private requiredId(token: string): number {
    const id: number | undefined = this.vocabulary.get(token);
    if (id === undefined) {
      throw new Error('词表缺少 ' + token);
    }
    return id;
  }

  private appendWord(word: string, output: number[]): void {
    if (word.length === 0) {
      return;
    }
    const chars: string[] = Array.from(word);
    if (chars.length > 100) {
      output.push(this.unknown);
      return;
    }
    const pieces: number[] = [];
    let start: number = 0;
    while (start < chars.length) {
      let end: number = chars.length;
      let id: number | undefined = undefined;
      while (end > start) {
        const part: string = chars.slice(start, end).join('');
        id = this.vocabulary.get(start === 0 ? part : '##' + part);
        if (id !== undefined) {
          break;
        }
        end--;
      }
      if (id === undefined) {
        output.push(this.unknown);
        return;
      }
      pieces.push(id);
      start = end;
    }
    for (const id of pieces) {
      output.push(id);
    }
  }

  private isPunctuation(code: number, char: string): boolean {
    return (code >= 33 && code <= 47) || (code >= 58 && code <= 64) ||
      (code >= 91 && code <= 96) || (code >= 123 && code <= 126) || /\p{P}/u.test(char);
  }

  private isChinese(code: number): boolean {
    return (code >= 0x4E00 && code <= 0x9FFF) || (code >= 0x3400 && code <= 0x4DBF) ||
      (code >= 0x20000 && code <= 0x2A6DF) || (code >= 0x2A700 && code <= 0x2B73F) ||
      (code >= 0x2B740 && code <= 0x2B81F) || (code >= 0x2B820 && code <= 0x2CEAF) ||
      (code >= 0xF900 && code <= 0xFAFF) || (code >= 0x2F800 && code <= 0x2FA1F);
  }
}
