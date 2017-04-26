uniform sampler2D texture1;
uniform sampler2D texture2;

uniform float offX;
uniform float offY;

void main() 
{
    vec4 px1 = texture2D(texture1, gl_TexCoord[0].xy);
    vec4 px2 = texture2D(texture2, gl_TexCoord[0].xy + vec2(offX, offY));

    gl_FragColor = vec4(abs(px1 - px2).rgb, 1.0);
}
