<template>
  <input type="file" ref="input" accept=".json" @change="handle">
</template>

<style scoped>
  input {
    display: none;
  }
</style>

<script>
export default {
  props: ['value'],

  methods: {
    /** Opens the file selection popup. */
    open() {
      console.log('Opening the file input.')
      this.$refs.input.click()
    },

    /** Reads a new JSON input file. */
    handle(e) {
      let reader = new FileReader()

      reader.onload = (e) => {
        let json = JSON.parse(e.target.result)
        console.log('Found JSON object:', json)
        this.$emit('input', json)
      }

      reader.readAsText(e.target.files[0])
    }
  }
}
</script>