import Foundation
import Metal

final class ZeroCopyBuffer: @unchecked Sendable {
    let pointer: UnsafeMutableRawPointer
    let count: Int
    let byteSize: Int
    let buffer: MTLBuffer
    private let deallocator: () -> Void

    static func allocate<T>(count: Int, dtype: T.Type) -> ZeroCopyBuffer? {
        let byteSize = count * MemoryLayout<T>.stride
        let pageSize = 16384  // vm_page_size on Apple Silicon (16KB)
        let alignedSize = ((byteSize + pageSize - 1) / pageSize) * pageSize

        var address: vm_address_t = 0
        let kr = vm_allocate(mach_task_self_, &address, vm_size_t(alignedSize), VM_FLAGS_ANYWHERE)
        guard kr == KERN_SUCCESS, address != 0 else { return nil }

        let ptr = UnsafeMutableRawPointer(bitPattern: address)!
        memset(ptr, 0, alignedSize)

        let device = MTLCreateSystemDefaultDevice()!
        let capturedAddress = address
        let capturedAlignedSize = vm_size_t(alignedSize)
        let buffer = device.makeBuffer(
            bytesNoCopy: ptr,
            length: byteSize,
            options: .storageModeShared,
            deallocator: { _, _ in vm_deallocate(mach_task_self_, capturedAddress, capturedAlignedSize) }
        )
        guard let buffer else {
            vm_deallocate(mach_task_self_, address, vm_size_t(alignedSize))
            return nil
        }

        return ZeroCopyBuffer(pointer: ptr, count: count, byteSize: byteSize, buffer: buffer, deallocator: {})
    }

    static func wrapExisting<T>(pointer: UnsafeMutableRawPointer, count: Int, dtype: T.Type, device: MTLDevice) -> MTLBuffer? {
        let byteSize = count * MemoryLayout<T>.stride
        return device.makeBuffer(
            bytesNoCopy: pointer,
            length: byteSize,
            options: .storageModeShared,
            deallocator: nil
        )
    }

    func asUnsafePointer<T>() -> UnsafePointer<T> {
        UnsafePointer(pointer.assumingMemoryBound(to: T.self))
    }

    func asUnsafeMutablePointer<T>() -> UnsafeMutablePointer<T> {
        pointer.assumingMemoryBound(to: T.self)
    }

    private init(pointer: UnsafeMutableRawPointer, count: Int, byteSize: Int, buffer: MTLBuffer, deallocator: @escaping () -> Void) {
        self.pointer = pointer
        self.count = count
        self.byteSize = byteSize
        self.buffer = buffer
        self.deallocator = deallocator
    }

    deinit {
        deallocator()
    }
}
