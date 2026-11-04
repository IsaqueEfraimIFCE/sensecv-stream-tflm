# Streaming TFLM harness: a single source file that talks the UART protocol.
idf_component_register(
    SRCS main.cc
    REQUIRES driver esp_timer
    PRIV_REQUIRES espressif__esp-tflite-micro
)
