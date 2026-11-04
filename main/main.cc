// SenseCV CBA E4 — ESP32-S3 TFLM streaming harness.
//
// One firmware for every config: the host streams the int8 .tflite model and then
// the int8 input frames over UART0 (the CH343/COM12 bridge); the device runs TFLM
// Invoke() per frame and returns the two raw int8 softmax heads + esp_timer latency.
// No re-quantisation on device: the host sends the exact int8 input bytes.
//
// Console/logs are routed to USB-Serial-JTAG (sdkconfig) so UART0 is a clean binary
// channel owned by the UART driver here.
//
// Wire protocol (UART0, 1500000 8N1, little-endian). Host -> device opcodes:
//   'P'                      ping            -> RESP 'R'
//   'M' u32 len  <len bytes> load model      -> RESP 'M' (see below) or 'E'
//   'I' <input_bytes bytes>  run one frame    -> RESP 'I' (see below) or 'E'
// Every response starts with the 4-byte magic {0xA5,0x5A,0xC3,0x3C} then a type byte:
//   'R' : ready, no payload
//   'M' : u32 arena_used, u32 input_bytes, u8 obs_classes(=2), u8 dev_classes(=3), u32 model_len
//   'I' : u32 latency_us, int8 obs[2], int8 dev[3]
//   'E' : u8 errcode
// The magic lets the host resync past ROM/boot noise on UART0.

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <cstring>
#include <stdint.h>

#include "driver/uart.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_task_wdt.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr uart_port_t kUart = UART_NUM_0;
constexpr int kBaud = 1500000;
constexpr int kRxBuf = 1 << 15;  // 32 KiB ISR ring buffer

// Tensor arena: 3 MiB in PSRAM was enough for all these models in the uint8 build;
// int8 activations are the same size. Allocated once at boot.
constexpr size_t kArenaSize = 3u * 1024 * 1024;

const uint8_t kMagic[4] = {0xA5, 0x5A, 0xC3, 0x3C};

enum Err : uint8_t {
  kErrModelSchema = 1,
  kErrArenaAlloc = 2,
  kErrModelAlloc = 3,
  kErrAllocTensors = 4,
  kErrOutputs = 5,
  kErrInputType = 6,
  kErrInvoke = 7,
  kErrShortRead = 8,
};

uint8_t* g_arena = nullptr;
uint8_t* g_model_buf = nullptr;
size_t g_model_cap = 0;

tflite::MicroInterpreter* g_interp = nullptr;
alignas(16) uint8_t g_interp_storage[sizeof(tflite::MicroInterpreter)];

TfLiteTensor* g_input = nullptr;
TfLiteTensor* g_out_obs = nullptr;  // 2-class head (obstaculo)
TfLiteTensor* g_out_dev = nullptr;  // 3-class head (desvio)

// ---- UART helpers -------------------------------------------------------

void uart_setup() {
  uart_config_t cfg = {};
  cfg.baud_rate = kBaud;
  cfg.data_bits = UART_DATA_8_BITS;
  cfg.parity = UART_PARITY_DISABLE;
  cfg.stop_bits = UART_STOP_BITS_1;
  cfg.flow_ctrl = UART_HW_FLOWCTRL_DISABLE;
  cfg.source_clk = UART_SCLK_DEFAULT;
  uart_driver_install(kUart, kRxBuf, 0, 0, nullptr, 0);
  uart_param_config(kUart, &cfg);
  uart_set_pin(kUart, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE,
               UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
}

// Blocking read of exactly n bytes. Returns true on success.
bool read_exact(uint8_t* dst, size_t n) {
  size_t got = 0;
  while (got < n) {
    int r = uart_read_bytes(kUart, dst + got, n - got, pdMS_TO_TICKS(20000));
    if (r <= 0) return false;
    got += static_cast<size_t>(r);
  }
  return true;
}

bool read_u32(uint32_t* v) {
  uint8_t b[4];
  if (!read_exact(b, 4)) return false;
  *v = static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) |
       (static_cast<uint32_t>(b[2]) << 16) | (static_cast<uint32_t>(b[3]) << 24);
  return true;
}

void write_all(const uint8_t* src, size_t n) {
  uart_write_bytes(kUart, reinterpret_cast<const char*>(src), n);
}

void put_u32(uint8_t* b, uint32_t v) {
  b[0] = v & 0xFF; b[1] = (v >> 8) & 0xFF; b[2] = (v >> 16) & 0xFF; b[3] = (v >> 24) & 0xFF;
}

void resp_header(uint8_t type) {
  write_all(kMagic, 4);
  write_all(&type, 1);
}

void resp_error(uint8_t code) {
  resp_header('E');
  write_all(&code, 1);
}

// ---- Model loading ------------------------------------------------------

bool build_resolver_and_interp(const uint8_t* model_data) {
  const tflite::Model* model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    resp_error(kErrModelSchema);
    return false;
  }

  // Covers CONV_2D, DEPTHWISE_CONV_2D, MEAN, FULLY_CONNECTED, SOFTMAX (plus spares).
  static tflite::MicroMutableOpResolver<8> resolver;
  static bool resolver_built = false;
  if (!resolver_built) {
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMean();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddQuantize();
    resolver.AddAveragePool2D();
    resolver_built = true;
  }

  if (g_interp != nullptr) {
    g_interp->~MicroInterpreter();
    g_interp = nullptr;
  }
  g_interp = new (g_interp_storage)
      tflite::MicroInterpreter(model, resolver, g_arena, kArenaSize);

  if (g_interp->AllocateTensors() != kTfLiteOk) {
    resp_error(kErrAllocTensors);
    return false;
  }

  g_input = g_interp->input(0);
  if (g_input->type != kTfLiteInt8) {
    resp_error(kErrInputType);
    return false;
  }

  // Map the two output heads by class count (shape), not index.
  g_out_obs = nullptr;
  g_out_dev = nullptr;
  for (size_t i = 0; i < g_interp->outputs_size(); ++i) {
    TfLiteTensor* t = g_interp->output(i);
    int last = t->dims->data[t->dims->size - 1];
    if (last == 2) g_out_obs = t;
    else if (last == 3) g_out_dev = t;
  }
  if (g_out_obs == nullptr || g_out_dev == nullptr) {
    resp_error(kErrOutputs);
    return false;
  }
  return true;
}

void cmd_load_model() {
  uint32_t len = 0;
  if (!read_u32(&len)) { resp_error(kErrShortRead); return; }

  if (len > g_model_cap) {
    if (g_model_buf) heap_caps_free(g_model_buf);
    g_model_buf = static_cast<uint8_t*>(
        heap_caps_malloc(len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    g_model_cap = g_model_buf ? len : 0;
  }
  if (g_model_buf == nullptr) {
    resp_error(kErrModelAlloc);
    return;
  }
  if (!read_exact(g_model_buf, len)) { resp_error(kErrShortRead); return; }

  if (!build_resolver_and_interp(g_model_buf)) return;  // error already sent

  uint8_t payload[14];
  put_u32(payload + 0, static_cast<uint32_t>(g_interp->arena_used_bytes()));
  put_u32(payload + 4, static_cast<uint32_t>(g_input->bytes));
  payload[8] = 2;  // obs classes
  payload[9] = 3;  // dev classes
  put_u32(payload + 10, len);
  resp_header('M');
  write_all(payload, sizeof(payload));
}

void cmd_infer() {
  if (g_interp == nullptr) { resp_error(kErrInvoke); return; }
  const size_t n = g_input->bytes;
  if (!read_exact(reinterpret_cast<uint8_t*>(g_input->data.int8), n)) {
    resp_error(kErrShortRead);
    return;
  }
  int64_t t0 = esp_timer_get_time();
  TfLiteStatus s = g_interp->Invoke();
  int64_t t1 = esp_timer_get_time();
  if (s != kTfLiteOk) { resp_error(kErrInvoke); return; }

  uint8_t payload[4 + 2 + 3];
  put_u32(payload, static_cast<uint32_t>(t1 - t0));
  const int8_t* obs = g_out_obs->data.int8;
  const int8_t* dev = g_out_dev->data.int8;
  payload[4] = static_cast<uint8_t>(obs[0]);
  payload[5] = static_cast<uint8_t>(obs[1]);
  payload[6] = static_cast<uint8_t>(dev[0]);
  payload[7] = static_cast<uint8_t>(dev[1]);
  payload[8] = static_cast<uint8_t>(dev[2]);
  resp_header('I');
  write_all(payload, sizeof(payload));
}

}  // namespace

extern "C" void app_main(void) {
  esp_task_wdt_deinit();
  tflite::InitializeTarget();
  uart_setup();

  g_arena = static_cast<uint8_t*>(
      heap_caps_malloc(kArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));

  // announce readiness (host waits for this, discarding any boot noise)
  resp_header('R');
  if (g_arena == nullptr) {
    resp_error(kErrArenaAlloc);
  }

  uint8_t op = 0;
  while (true) {
    if (!read_exact(&op, 1)) continue;  // timeout: keep waiting
    switch (op) {
      case 'P': resp_header('R'); break;
      case 'M': cmd_load_model(); break;
      case 'I': cmd_infer(); break;
      default: break;  // ignore stray bytes
    }
  }
}
