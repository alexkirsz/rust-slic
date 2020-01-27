// Note that a dynamic `import` statement here is required due to
// webpack/webpack#6615, but in theory `import { greet } from './pkg/hello_world';`
// will work here one day as well!
const rust = import("./pkg/rust_slic");

async function setup() {
  const rust = await import("./pkg/rust_slic");

  rust.init_panic_hook();

  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/*";
  document.body.appendChild(input);

  const imgSrc = document.createElement("img");
  document.body.appendChild(imgSrc);

  const imgRes = document.createElement("img");
  document.body.appendChild(imgRes);

  input.addEventListener("change", e => {
    const selectedFile = input.files[0];

    const dataUrlReader = new FileReader();
    dataUrlReader.addEventListener("load", e => {
      imgSrc.src = dataUrlReader.result;
    });
    dataUrlReader.readAsDataURL(selectedFile);

    const arrayBufferReader = new FileReader();
    arrayBufferReader.addEventListener("load", e => {
      const bytes = new Uint8Array(arrayBufferReader.result);
      const slic = rust.WasmSLIC.new(bytes, 10, 20, 5.0);
      const res = slic.process(50.0, 10);
      const base64Data = btoa(String.fromCharCode.apply(null, res));
      imgRes.src = "data:image/png;base64," + base64Data;
    });
    arrayBufferReader.readAsArrayBuffer(new Blob([selectedFile]));
  });
}

setup().catch(console.error);
