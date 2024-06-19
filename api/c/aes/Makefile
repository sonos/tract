CC = gcc
CFLAGS = -Wall -Wextra -Wpedantic -Werror -g
#LDFLAGS = -I/home1/public/papafrkon/dAIEdgeServer/include -L/home1/public/papafrkon/dAIEdgeServer/lib -lmbedtls -lmbedx509 -lmbedcrypto
LDFLAGS = -I/home/konstantina/Desktop/CARV/dAIEdge/inference/dAIEdgeServer/include -L/home/konstantina/Desktop/CARV/dAIEdge/inference/dAIEdgeServer/lib -lmbedtls -lmbedx509 -lmbedcrypto

all: aes gcm gcm_test

aes: aes.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

gcm: gcm.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

gcm_test: gcm_test.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f aes gcm gcm_test
