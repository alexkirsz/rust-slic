import React, { useState, useEffect, useRef } from "react";
import ReactDOM from "react-dom";
import {
  Slider,
  Typography,
  CssBaseline,
  Box,
  Input,
  Paper,
  Button,
  Grid,
  Switch
} from "@material-ui/core";

function App({ rust }) {
  const [m, setM] = useState(10);
  const [s, setS] = useState(20);
  const [errorThreshold, setErrorThreshold] = useState(50);
  const [minSize, setMinSize] = useState(10);
  const [textureCoeff, setTextureCoeff] = useState(10);
  const [c8, setC8] = useState(true);
  const [srcImgBuffer, setSrcImgBuffer] = useState(null);
  const [srcImg, setSrcImg] = useState(null);
  const [resImg, setResImg] = useState("");
  const [slic, setSlic] = useState(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (srcImgBuffer == null) {
      return;
    }

    const slic = rust.WasmSLIC.new(srcImgBuffer);
    setSlic(slic);
  }, [srcImgBuffer]);

  useEffect(() => {
    if (slic == null) {
      return;
    }

    const timeout = setTimeout(() => {
      if (slic == null) {
        return;
      }
      const res = slic.process(m, s, textureCoeff, errorThreshold, minSize, c8);
      const base64Data = btoa(String.fromCharCode.apply(null, res));
      setResImg("data:image/png;base64," + base64Data);
    }, 1000);

    return () => clearTimeout(timeout);
  }, [m, s, errorThreshold, minSize, textureCoeff, c8, slic]);

  return (
    <>
      <CssBaseline />
      <Box display="flex" padding={2} height="100%">
        <Grid container spacing={2}>
          <Grid item sm={6} xs={12}>
            <Box position="absolute">
              <Input
                type="file"
                inputRef={inputRef}
                style={{ visibility: "hidden" }}
                onChange={e => {
                  const selectedFile = e.target.files[0];

                  const dataUrlReader = new FileReader();
                  dataUrlReader.addEventListener("load", e => {
                    setSrcImg(dataUrlReader.result);
                  });
                  dataUrlReader.readAsDataURL(selectedFile);

                  const arrayBufferReader = new FileReader();
                  arrayBufferReader.addEventListener("load", e => {
                    const bytes = new Uint8Array(arrayBufferReader.result);
                    setSrcImgBuffer(bytes);
                  });
                  arrayBufferReader.readAsArrayBuffer(new Blob([selectedFile]));
                }}
              />
            </Box>

            <Box
              display="flex"
              width="100%"
              height="100%"
              alignItems="center"
              justifyContent="center"
              position="relative"
            >
              {srcImg != null && (
                <img
                  src={srcImg}
                  style={{
                    objectFit: "contain",
                    width: "100%",
                    height: "auto"
                  }}
                />
              )}
              <Box
                {...(srcImg != null
                  ? { position: "absolute", top: 0, margin: "0 auto" }
                  : {})}
              >
                <Button
                  color="primary"
                  variant="contained"
                  onClick={() => inputRef.current.click()}
                >
                  Select an image
                </Button>
              </Box>
            </Box>
          </Grid>

          <Grid item sm={6} xs={12}>
            <Box
              display="flex"
              width="100%"
              height="100%"
              alignItems="center"
              justifyContent="center"
            >
              <img
                src={resImg}
                style={{ objectFit: "contain", width: "100%", height: "auto" }}
              />
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Paper variant="outlined">
              <Box p={2}>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography gutterBottom variant="caption">
                      M
                    </Typography>
                    <Slider
                      value={m}
                      valueLabelDisplay="auto"
                      step={1}
                      min={1}
                      max={100}
                      onChange={(e, nv) => setM(nv)}
                    />
                    <Typography gutterBottom variant="caption">
                      S
                    </Typography>
                    <Slider
                      value={s}
                      valueLabelDisplay="auto"
                      step={1}
                      min={1}
                      max={100}
                      onChange={(e, nv) => setS(nv)}
                    />
                  </Grid>

                  <Grid item xs={6}>
                    <Typography gutterBottom variant="caption">
                      Region Min Size
                    </Typography>
                    <Slider
                      value={minSize}
                      valueLabelDisplay="auto"
                      step={1}
                      min={5}
                      max={1000}
                      onChange={(e, nv) => setMinSize(nv)}
                    />
                    <Typography gutterBottom variant="caption">
                      Texture Coefficient
                    </Typography>
                    <Slider
                      value={textureCoeff}
                      valueLabelDisplay="auto"
                      step={0.1}
                      min={1}
                      max={100}
                      onChange={(e, nv) => setTextureCoeff(nv)}
                    />
                  </Grid>

                  <Grid item xs={6}>
                    <Typography gutterBottom variant="caption">
                      Error Threshold
                    </Typography>
                    <Slider
                      value={errorThreshold}
                      valueLabelDisplay="auto"
                      step={0.1}
                      min={10}
                      max={500}
                      onChange={(e, nv) => setErrorThreshold(nv)}
                    />
                  </Grid>

                  <Grid item xs={6}>
                    <Typography gutterBottom variant="caption">
                      Connexity
                    </Typography>
                    <Grid
                      component="label"
                      container
                      alignItems="center"
                      spacing={1}
                    >
                      <Grid item>4</Grid>
                      <Grid item>
                        <Switch checked={c8} onChange={(e, nv) => setC8(nv)} />
                      </Grid>
                      <Grid item>8</Grid>
                    </Grid>
                  </Grid>
                </Grid>
              </Box>
            </Paper>
          </Grid>
        </Grid>
      </Box>
    </>
  );
}

async function setup() {
  const rust = await import("./pkg/rust_slic");
  rust.init_panic_hook();

  ReactDOM.render(<App rust={rust} />, document.getElementById("app"));
}

setup().catch(console.error);
