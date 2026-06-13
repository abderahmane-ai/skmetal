import Metal

actor BufferPool {
    private var available: [Int: [ZeroCopyBuffer]] = [:]
    private let device: MTLDevice

    init(device: MTLDevice) { self.device = device }

    func acquire(count: Int, dtype: Float.Type) -> ZeroCopyBuffer? {
        let key = count * MemoryLayout<Float>.stride
        if var buffers = available[key], !buffers.isEmpty {
            return buffers.removeLast()
        }
        return ZeroCopyBuffer.allocate(count: count, dtype: Float.self)
    }

    func release(_ buffer: ZeroCopyBuffer) {
        let key = buffer.byteSize
        available[key, default: []].append(buffer)
    }

    func prewarm(sizes: [Int]) {
        for size in sizes {
            if let buf = acquire(count: size, dtype: Float.self) {
                release(buf)
            }
        }
    }
}