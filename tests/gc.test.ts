import { describe, it, expect, beforeAll } from 'vitest';
import eig from '../dist/index.mjs';

describe('Garbage Collector (GC)', () => {
  beforeAll(async () => {
    await eig.ready;
  });

  it('tracks allocations and flushes unreferenced objects', () => {
    const M1 = new eig.Matrix(10, 10);
    const M2 = new eig.Matrix(10, 10);
    expect(eig.GC.objects.size).toBeGreaterThanOrEqual(2);

    const flushed = eig.GC.flush();
    expect(flushed).toBeGreaterThanOrEqual(2);
    expect(eig.GC.objects.size).toBe(0);
  });

  it('protects objects registered with pushException', () => {
    const M1 = new eig.Matrix([[1, 2], [3, 4]]);
    const M2 = new eig.Matrix([[5, 6], [7, 8]]);

    eig.GC.pushException(M1);

    // M2 should be flushed, M1 should remain
    const flushed = eig.GC.flush();
    expect(flushed).toBeGreaterThanOrEqual(1);
    expect(eig.GC.objects.has(M1)).toBe(true);
    expect(M1.get(0, 0)).toBe(1);

    // Pop exception and flush again
    eig.GC.popException(M1);
    const flushedFinal = eig.GC.flush();
    expect(flushedFinal).toBe(1);
    expect(eig.GC.objects.has(M1)).toBe(false);
  });
});
