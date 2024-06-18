CC = gcc
CFLAGS = -Wall -Wextra -Wpedantic -Werror -g
LDFLAGS = -I/home1/public/papafrkon/dAIEdgeServer/include -L/home1/public/papafrkon/dAIEdgeServer/lib -lmbedtls -lmbedx509 -lmbedcrypto

all: aes

aes: aes.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f aes
