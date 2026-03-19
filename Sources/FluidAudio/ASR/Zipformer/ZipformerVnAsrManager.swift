import Accelerate
import Foundation
import OnnxRuntimeBindings

public actor ZipformerVnAsrManager {
    private let logger = AppLogger(category: "ZipformerVnAsrManager")
    private let audioConverter = AudioConverter()
    private let melExtractor = ZipformerVnMelSpectrogram()

    private var models: ZipformerVnAsrModels?

    public init() {}

    public var isReady: Bool {
        models != nil
    }

    public func loadModels(from directory: URL) throws {
        models = try ZipformerVnAsrModels.load(from: directory)
    }

    public func loadModels(directory: URL? = nil, forceDownload: Bool = false) async throws {
        models = try await ZipformerVnAsrModels.downloadAndLoad(to: directory, force: forceDownload)
    }

    public func transcribe(url: URL) async throws -> ASRResult {
        let samples: [Float]
        do {
            samples = try audioConverter.resampleAudioFile(url)
        } catch {
            throw ASRError.fileAccessFailed(url, error)
        }
        return try await transcribe(audioSamples: samples)
    }

    public func transcribe(audioSamples: [Float]) async throws -> ASRResult {
        guard let models else {
            throw ASRError.notInitialized
        }

        guard !audioSamples.isEmpty else {
            throw ASRError.invalidAudioData
        }

        let start = CFAbsoluteTimeGetCurrent()

        let (features, frameCount) = melExtractor.compute(audio: audioSamples)
        let encoderOut = try runEncoder(session: models.encoderSession, features: features, frameCount: frameCount)
        let tokenIds = try runGreedyDecode(
            encoderFrames: encoderOut.frames,
            validFrameCount: encoderOut.validFrameCount,
            decoderSession: models.decoderSession,
            joinerSession: models.joinerSession
        )

        let text = detokenize(tokenIds: tokenIds, tokens: models.tokens)
        let processingTime = CFAbsoluteTimeGetCurrent() - start
        let duration = Double(audioSamples.count) / Double(ASRConstants.sampleRate)

        let durationStr = String(format: "%.1f", duration)
        let procStr = String(format: "%.2f", processingTime)
        logger.info("Zipformer transcribed \(durationStr)s in \(procStr)s")

        return ASRResult(
            text: text,
            confidence: tokenIds.isEmpty ? 0 : 1,
            duration: duration,
            processingTime: processingTime,
            tokenTimings: nil
        )
    }

    private func runEncoder(session: ORTSession, features: [Float], frameCount: Int) throws -> (frames: [Float], validFrameCount: Int) {
        let xData = NSMutableData(data: Data(bytes: features, count: features.count * MemoryLayout<Float>.size))
        let xValue = try ORTValue(
            tensorData: xData,
            elementType: ORTTensorElementDataType.float,
            shape: [1, NSNumber(value: frameCount), 80]
        )

        var lensValueRaw = [Int64(frameCount)]
        let xLensData = NSMutableData(data: Data(bytes: &lensValueRaw, count: MemoryLayout<Int64>.size))
        let xLensValue = try ORTValue(
            tensorData: xLensData,
            elementType: ORTTensorElementDataType.int64,
            shape: [1]
        )

        let outputs = try session.run(
            withInputs: ["x": xValue, "x_lens": xLensValue],
            outputNames: Set(["encoder_out", "encoder_out_lens"]),
            runOptions: nil
        )

        guard let encoderValue = outputs["encoder_out"], let lensValue = outputs["encoder_out_lens"] else {
            throw ASRError.processingFailed("Zipformer encoder outputs missing")
        }

    let encoderData = Data(referencing: try encoderValue.tensorData())
    let lensData = Data(referencing: try lensValue.tensorData())

        let frames = encoderData.withUnsafeBytes { raw -> [Float] in
            let ptr = raw.bindMemory(to: Float.self)
            return Array(ptr)
        }

        let validFrames = lensData.withUnsafeBytes { raw -> Int in
            let ptr = raw.bindMemory(to: Int64.self)
            return ptr.isEmpty ? frameCount : Int(ptr[0])
        }

        return (frames, min(validFrames, frameCount))
    }

    private func runGreedyDecode(
        encoderFrames: [Float],
        validFrameCount: Int,
        decoderSession: ORTSession,
        joinerSession: ORTSession
    ) throws -> [Int] {
        let encoderDim = 512
        guard encoderFrames.count >= validFrameCount * encoderDim else {
            throw ASRError.processingFailed("Zipformer encoder output shape mismatch")
        }

        let blankId = 0
        var context: [Int64] = [0, 0]
        var tokenIds: [Int] = []

        for t in 0..<validFrameCount {
            let frameBase = t * encoderDim
            let frame = Array(encoderFrames[frameBase..<(frameBase + encoderDim)])

            var emitted = true
            var emittedCount = 0
            while emitted && emittedCount < 4 {
                emitted = false

                let decoderOut = try runDecoder(session: decoderSession, context: context)
                let logits = try runJoiner(session: joinerSession, encoderFrame: frame, decoderOut: decoderOut)
                let best = argmax(logits)

                if best != blankId {
                    tokenIds.append(best)
                    context[0] = context[1]
                    context[1] = Int64(best)
                    emitted = true
                    emittedCount += 1
                }
            }
        }

        return tokenIds
    }

    private func runDecoder(session: ORTSession, context: [Int64]) throws -> [Float] {
        var contextVar = context
        let data = NSMutableData(data: Data(bytes: &contextVar, count: contextVar.count * MemoryLayout<Int64>.size))
        let value = try ORTValue(
            tensorData: data,
            elementType: ORTTensorElementDataType.int64,
            shape: [1, NSNumber(value: context.count)]
        )

        let outputs = try session.run(
            withInputs: ["y": value],
            outputNames: Set(["decoder_out"]),
            runOptions: nil
        )

        guard let decoderValue = outputs["decoder_out"] else {
            throw ASRError.processingFailed("Zipformer decoder output missing")
        }

        let outData = Data(referencing: try decoderValue.tensorData())
        return outData.withUnsafeBytes { raw -> [Float] in
            let ptr = raw.bindMemory(to: Float.self)
            return Array(ptr)
        }
    }

    private func runJoiner(session: ORTSession, encoderFrame: [Float], decoderOut: [Float]) throws -> [Float] {
        var enc = encoderFrame
        var dec = decoderOut

        let encData = NSMutableData(data: Data(bytes: &enc, count: enc.count * MemoryLayout<Float>.size))
        let decData = NSMutableData(data: Data(bytes: &dec, count: dec.count * MemoryLayout<Float>.size))

        let encValue = try ORTValue(
            tensorData: encData,
            elementType: ORTTensorElementDataType.float,
            shape: [1, NSNumber(value: enc.count)]
        )

        let decValue = try ORTValue(
            tensorData: decData,
            elementType: ORTTensorElementDataType.float,
            shape: [1, NSNumber(value: dec.count)]
        )

        let outputs = try session.run(
            withInputs: ["encoder_out": encValue, "decoder_out": decValue],
            outputNames: Set(["logit"]),
            runOptions: nil
        )

        guard let logitValue = outputs["logit"] else {
            throw ASRError.processingFailed("Zipformer joiner output missing")
        }

        let logitData = Data(referencing: try logitValue.tensorData())
        return logitData.withUnsafeBytes { raw -> [Float] in
            let ptr = raw.bindMemory(to: Float.self)
            return Array(ptr)
        }
    }

    private func argmax(_ values: [Float]) -> Int {
        guard var best = values.first else {
            return 0
        }
        var bestIdx = 0
        for i in 1..<values.count {
            if values[i] > best {
                best = values[i]
                bestIdx = i
            }
        }
        return bestIdx
    }

    private func detokenize(tokenIds: [Int], tokens: [Int: String]) -> String {
        var pieces: [String] = []
        pieces.reserveCapacity(tokenIds.count)

        for id in tokenIds {
            guard let token = tokens[id] else {
                continue
            }
            if token == "<blk>" || token == "<sos/eos>" || token == "<unk>" {
                continue
            }
            pieces.append(token)
        }

        let merged = pieces.joined()
            .replacingOccurrences(of: "▁", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        return merged
    }
}

private final class ZipformerVnMelSpectrogram {
    private let nFFT = 512
    private let hopLength = 160
    private let winLength = 400
    private let nMels = 80
    private let sampleRate = 16000

    private let window: [Float]
    private let filterBank: [Float]
    private var fftSetup: vDSP_DFT_Setup?

    private var realIn: [Float]
    private var imagIn: [Float]
    private var realOut: [Float]
    private var imagOut: [Float]
    private var power: [Float]

    init() {
        self.window = ZipformerVnMelSpectrogram.makeHann(length: winLength)
        self.filterBank = ZipformerVnMelSpectrogram.makeMelFilterBank(
            nFFT: nFFT,
            nMels: nMels,
            sampleRate: sampleRate,
            fMin: 0,
            fMax: Float(sampleRate / 2)
        )
        self.realIn = [Float](repeating: 0, count: nFFT)
        self.imagIn = [Float](repeating: 0, count: nFFT)
        self.realOut = [Float](repeating: 0, count: nFFT)
        self.imagOut = [Float](repeating: 0, count: nFFT)
        self.power = [Float](repeating: 0, count: nFFT / 2 + 1)
        self.fftSetup = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(nFFT), .FORWARD)
    }

    deinit {
        if let fftSetup {
            vDSP_DFT_DestroySetup(fftSetup)
        }
    }

    func compute(audio: [Float]) -> (features: [Float], frameCount: Int) {
        let paddedCount = audio.count + nFFT
        var padded = [Float](repeating: 0, count: paddedCount)
        let pad = nFFT / 2
        if !audio.isEmpty {
            padded.replaceSubrange(pad..<(pad + audio.count), with: audio)
        }

        let frameCount = max(1, 1 + (paddedCount - winLength) / hopLength)
        var output = [Float](repeating: 0, count: frameCount * nMels)
        let freqBins = nFFT / 2 + 1

        var frame = [Float](repeating: 0, count: nFFT)
        var melFrame = [Float](repeating: 0, count: nMels)

        for t in 0..<frameCount {
            let start = t * hopLength
            frame.withUnsafeMutableBufferPointer { ptr in
                ptr.initialize(repeating: 0)
            }

            let available = max(0, min(winLength, paddedCount - start))
            if available > 0 {
                for i in 0..<available {
                    frame[i] = padded[start + i] * window[i]
                }
            }

            if let fftSetup {
                vDSP_DFT_Execute(fftSetup, frame, imagIn, &realOut, &imagOut)
            }

            for i in 0..<freqBins {
                let r = realOut[i]
                let im = imagOut[i]
                power[i] = r * r + im * im
            }

            filterBank.withUnsafeBufferPointer { fPtr in
                power.withUnsafeBufferPointer { pPtr in
                    melFrame.withUnsafeMutableBufferPointer { oPtr in
                        vDSP_mmul(
                            fPtr.baseAddress!, 1,
                            pPtr.baseAddress!, 1,
                            oPtr.baseAddress!, 1,
                            vDSP_Length(nMels),
                            1,
                            vDSP_Length(freqBins)
                        )
                    }
                }
            }

            for m in 0..<nMels {
                output[t * nMels + m] = logf(max(melFrame[m], 1e-10))
            }
        }

        return (output, frameCount)
    }

    private static func makeHann(length: Int) -> [Float] {
        (0..<length).map { i in
            0.5 * (1 - cosf(2 * .pi * Float(i) / Float(length - 1)))
        }
    }

    private static func makeMelFilterBank(
        nFFT: Int,
        nMels: Int,
        sampleRate: Int,
        fMin: Float,
        fMax: Float
    ) -> [Float] {
        let numBins = nFFT / 2 + 1

        func hzToMel(_ hz: Float) -> Float {
            2595 * log10f(1 + hz / 700)
        }

        func melToHz(_ mel: Float) -> Float {
            700 * (powf(10, mel / 2595) - 1)
        }

        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)
        let melPoints = (0..<(nMels + 2)).map { i in
            melMin + (Float(i) / Float(nMels + 1)) * (melMax - melMin)
        }

        let hzPoints = melPoints.map(melToHz)
        let bins = hzPoints.map { hz in
            Int(floor((Float(nFFT + 1) * hz) / Float(sampleRate)))
        }

        var filterBank = [Float](repeating: 0, count: nMels * numBins)
        for m in 1...nMels {
            let left = max(0, bins[m - 1])
            let center = max(left + 1, bins[m])
            let right = max(center + 1, min(numBins - 1, bins[m + 1]))

            if center > left {
                for k in left..<center where k < numBins {
                    filterBank[(m - 1) * numBins + k] = Float(k - left) / Float(center - left)
                }
            }

            if right > center {
                for k in center..<right where k < numBins {
                    filterBank[(m - 1) * numBins + k] = Float(right - k) / Float(right - center)
                }
            }
        }

        return filterBank
    }
}
