<template>
  <div>
    <v-card-text
      class="mt-3"
      style="text-align: center; padding: 0">

      <span v-if="!isOnly">Unknown (depends on input)</span>
      <span v-else-if="shape.length == 0">{{ content[0] }}</span>
      <span v-else-if="isSmall" style="white-space: pre;">{{ matrix.toString() }}</span>
      <v-btn
        large color="primary"
        v-else
        @click="displayDialog = true">
        Display
      </v-btn>
    </v-card-text>
    <v-dialog v-model="displayDialog" max-width="850px">
      <v-card>
        <v-card-title>
          <span>Edge value:</span>
        </v-card-title>
        <v-card-text
          style="text-align: center; white-space: pre;">{{ matrix.toString() }}</v-card-text>
      </v-card>
    </v-dialog>
  </div>
</template>

<script>
  import numjs from 'numjs'

  numjs.config.printThreshold = 7;

  export default {
    props: ['value'],

    data: () => ({
      displayDialog: false
    }),

    computed: {
      isOnly() {
        return !!this.value.Only
      },

      isSmall() {
        return this.shape.every(d => d <= numjs.config.printThreshold)
      },

      shape() {
        return this.value.Only[1]
      },

      content() {
        return this.value.Only[2]
      },

      matrix() {
        return numjs.array(this.content).reshape(this.shape)
      },
    }
  }
</script>
