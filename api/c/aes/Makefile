CC = gcc
CFLAGS = -Wall -Wextra -Wpedantic -Werror -g
#LDFLAGS = -I/home1/public/papafrkon/dAIEdgeServer/include -L/home1/public/papafrkon/dAIEdgeServer/lib -lmbedtls -lmbedx509 -lmbedcrypto
LDFLAGS = -I/home/konstantina/Desktop/CARV/dAIEdge/inference/dAIEdgeServer/include -I./include -L/home/konstantina/Desktop/CARV/dAIEdge/inference/dAIEdgeServer/lib -lmbedtls -lmbedx509 -lmbedcrypto

all: cbc_final gcm

cbc_final: cbc/cbc.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

gcm: gcm_first gcm_second

gcm_first: gcm/gcm.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

gcm_second: gcm/gcm_test.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f cbc_final gcm_first gcm_second
