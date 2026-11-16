import { common } from '@kit.AbilityKit';
import { LengthMetrics } from '@kit.ArkUI';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { PermissionModel, InferenceResult } from '../model/PermissionModel';

const INK: string = '#142E26';
const MUTED: string = '#4E665A';
const GREEN: string = '#174B3C';
const CANVAS: string = '#F3F6EF';
const LINE: string = '#DFE7DB';
const WHITE: string = '#FFFFFF';

class PermissionLabel {
  title: string;
  android: string;
  harmony: string;
  constructor(title: string, android: string, harmony: string) {
    this.title = title;
    this.android = android;
    this.harmony = harmony;
  }
}

const LABELS: PermissionLabel[] = [
  new PermissionLabel('摄像头', 'android.permission.CAMERA', 'ohos.permission.CAMERA'),
  new PermissionLabel('麦克风', 'android.permission.RECORD_AUDIO', 'ohos.permission.MICROPHONE'),
  new PermissionLabel('粗略定位', 'android.permission.ACCESS_COARSE_LOCATION', 'ohos.permission.APPROXIMATELY_LOCATION'),
  new PermissionLabel('精确定位', 'android.permission.ACCESS_FINE_LOCATION', 'ohos.permission.LOCATION')
];

@Entry
@Component
struct Index {
  @State private instruction: string = '录一段视频，别录声音';
  @State private submitted: string = '';
  @State private busy: boolean = false;
  @State private ready: boolean = false;
  @State private hasResult: boolean = false;
  @State private unknown: boolean = false;
  @State private showHarmony: boolean = false;
  @State private elapsedMs: number = 0;
  @State private errorMessage: string = '';
  @State private simulation: string = '';
  @State private selected: number[] = [];
  @State private backend: string = '';
  @State private benchmarkReport: string = '';
  @State private diagnostic: string = '';
  private readonly model: PermissionModel = new PermissionModel();

  aboutToAppear(): void {
    this.initialize();
  }

  private async initialize(): Promise<void> {
    if (this.ready || this.busy) {
      return;
    }
    this.busy = true;
    this.errorMessage = '';
    try {
      const context: common.UIAbilityContext = this.getUIContext().getHostContext() as common.UIAbilityContext;
      await this.model.load(context.resourceManager);
      this.ready = true;
    } catch (error) {
      this.errorMessage = (error as Error).message || '模型加载失败。';
      hilog.error(0xFF00, 'PermissionNPU', '[PNPU][ERROR] initialize: %{public}s', this.errorMessage);
    } finally {
      this.diagnostic = this.model.diagnostic;
      this.busy = false;
    }
  }

  private edit(value: string): void {
    this.instruction = value;
    this.hasResult = false;
    this.simulation = '';
    this.errorMessage = '';
    this.benchmarkReport = '';
  }

  private async recognize(): Promise<void> {
    const text: string = this.instruction.trim();
    if (text.length === 0 || this.busy || !this.ready) {
      return;
    }
    this.busy = true;
    this.hasResult = false;
    this.errorMessage = '';
    this.simulation = '';
    try {
      const result: InferenceResult = await this.model.predict(text);
      const selected: number[] = [];
      for (let i = 0; i < LABELS.length; i++) {
        if (result.scores[i] >= result.thresholds[i]) {
          selected.push(i);
        }
      }
      this.selected = selected;
      this.unknown = result.scores[4] >= result.thresholds[4];
      this.elapsedMs = result.elapsedMs;
      this.backend = result.backend;
      this.submitted = text;
      this.hasResult = true;
    } catch (error) {
      const failure: Error = error as Error;
      this.errorMessage = failure.message || '推理失败，请重试。';
      hilog.error(0xFF00, 'PermissionNPU', '[PNPU][ERROR] recognize: %{public}s', this.errorMessage);
    } finally {
      this.diagnostic = this.model.diagnostic;
      this.busy = false;
    }
  }

  private async runBenchmark(): Promise<void> {
    if (this.busy || !this.ready) {
      return;
    }
    this.busy = true;
    this.benchmarkReport = '';
    this.errorMessage = '';
    try {
      this.benchmarkReport = await this.model.benchmark(this.instruction.trim());
    } catch (error) {
      this.errorMessage = (error as Error).message || '性能测试失败';
      hilog.error(0xFF00, 'PermissionNPU', '[PNPU][ERROR] benchmark: %{public}s', this.errorMessage);
    } finally {
      this.diagnostic = this.model.diagnostic;
      this.busy = false;
    }
  }

  private permissionName(index: number): string {
    if (!this.showHarmony) {
      return LABELS[index].android;
    }
    if (index === 3) {
      return LABELS[index].harmony + '\n' + LABELS[2].harmony;
    }
    return LABELS[index].harmony;
  }

  private simulateMicrophone(): void {
    if (this.unknown) {
      this.simulation = '无法判断模拟访问是否符合指令。';
    } else if (this.selected.includes(1)) {
      this.simulation = '模拟结果：本次指令涉及麦克风。';
    } else {
      this.simulation = '模拟结果：麦克风访问超出本次指令范围。';
    }
  }

  @Builder
  sample(title: string, text: string) {
    Button(title, { type: ButtonType.Capsule })
      .fontSize(13)
      .fontColor(GREEN)
      .backgroundColor('#E5EEE1')
      .height(36)
      .padding({ left: 12, right: 12 })
      .enabled(!this.busy)
      .onClick(() => this.edit(text))
  }

  @Builder
  platform(title: string, harmony: boolean) {
    Button(title, { type: ButtonType.Capsule })
      .fontSize(12)
      .fontColor(this.showHarmony === harmony ? WHITE : GREEN)
      .backgroundColor(this.showHarmony === harmony ? GREEN : '#E5EEE1')
      .height(32)
      .padding({ left: 12, right: 12 })
      .onClick(() => { this.showHarmony = harmony; })
  }

  build() {
    Scroll() {
      Column({ space: 18 }) {
        Column({ space: 8 }) {
          Text('权限意图')
            .fontSize(36)
            .fontWeight(FontWeight.Bold)
            .fontColor(INK)
          Text('从中文指令识别相机、麦克风和定位权限')
            .fontSize(15)
            .fontColor(MUTED)
          Text(this.ready ? 'MiniRBT · HiAI 优先' : (this.busy ? '正在加载模型' : '模型未就绪'))
            .fontSize(12)
            .fontWeight(FontWeight.Medium)
            .fontColor(GREEN)
            .padding({ top: 6, bottom: 6, left: 11, right: 11 })
            .backgroundColor('#DFEDDA')
            .borderRadius(14)
        }
        .width('100%')
        .alignItems(HorizontalAlign.Start)
        .padding({ top: 12, bottom: 6 })

        Column({ space: 12 }) {
          Text('用户指令')
            .fontSize(16)
            .fontWeight(FontWeight.Bold)
            .fontColor(INK)
          TextArea({ text: this.instruction, placeholder: '例如：录一段视频，别录声音' })
            .fontSize(17)
            .fontColor(INK)
            .backgroundColor('#F6F8F3')
            .borderRadius(12)
            .maxLength(240)
            .height(102)
            .width('100%')
            .enabled(!this.busy)
            .onChange((value: string) => this.edit(value))
          Flex({ wrap: FlexWrap.Wrap, space: { main: LengthMetrics.vp(6), cross: LengthMetrics.vp(6) } }) {
            this.sample('无声录像', '录一段视频，别录声音')
            this.sample('有声录像', '录一段带声音的视频')
            this.sample('大致位置', '用我的大致位置查天气')
          }
          Button(this.ready ? (this.busy ? '识别中…' : '识别权限') :
            (this.busy ? '正在加载模型…' : '重试加载'), { type: ButtonType.Normal })
            .fontSize(16)
            .fontWeight(FontWeight.Medium)
            .fontColor(WHITE)
            .backgroundColor(GREEN)
            .borderRadius(12)
            .width('100%')
            .height(48)
            .enabled(!this.busy && (!this.ready || this.instruction.trim().length > 0))
            .onClick(() => {
              if (this.ready) {
                this.recognize();
              } else {
                this.initialize();
              }
            })
          Button('测量推理时延')
            .fontSize(14)
            .fontColor(GREEN)
            .backgroundColor('#E5EEE1')
            .width('100%')
            .enabled(this.ready && !this.busy && this.instruction.trim().length > 0)
            .onClick(() => { this.runBenchmark(); })
          if (this.benchmarkReport.length > 0) {
            Text(this.benchmarkReport)
              .fontSize(13)
              .fontColor(INK)
              .width('100%')
          }
          if (this.errorMessage.length > 0) {
            Text(this.errorMessage)
              .fontSize(13)
              .fontColor('#9E3B31')
          }
        }
        .width('100%')
        .alignItems(HorizontalAlign.Start)
        .padding(18)
        .backgroundColor(WHITE)
        .border({ width: 1, color: LINE, radius: 18 })

        if (this.diagnostic.length > 0) {
          Text(this.diagnostic)
            .fontSize(12)
            .fontColor(MUTED)
            .width('100%')
            .wordBreak(WordBreak.BREAK_ALL)
        }

        if (this.hasResult) {
          Column({ space: 12 }) {
            Row() {
              Text('识别结果')
                .fontSize(16)
                .fontWeight(FontWeight.Bold)
                .fontColor(INK)
                .layoutWeight(1)
              Text(this.backend + ' · ' + this.elapsedMs.toString() + ' ms')
                .fontSize(12)
                .fontColor(MUTED)
            }
            .width('100%')
            Text(this.submitted)
              .fontSize(13)
              .fontColor(MUTED)
            Row({ space: 6 }) {
              this.platform('Android', false)
              this.platform('HarmonyOS', true)
            }
            if (this.unknown) {
              Text('无法确定。请明确要执行的操作及其访问范围。')
                .fontSize(15)
                .fontColor('#9E3B31')
            } else if (this.selected.length === 0) {
              Text('未识别到 Demo 覆盖的权限。')
                .fontSize(15)
                .fontColor(MUTED)
            } else {
              ForEach(this.selected, (index: number) => {
                Column({ space: 5 }) {
                  Text(LABELS[index].title)
                    .fontSize(15)
                    .fontWeight(FontWeight.Medium)
                    .fontColor(INK)
                  Text(this.permissionName(index))
                    .fontSize(12)
                    .fontColor(GREEN)
                    .width('100%')
                    .wordBreak(WordBreak.BREAK_ALL)
                }
                .alignItems(HorizontalAlign.Start)
                .width('100%')
                .padding(12)
                .backgroundColor('#EDF4E9')
                .borderRadius(10)
              }, (index: number) => index.toString())
            }
            Button('模拟麦克风访问', { type: ButtonType.Normal })
              .fontSize(13)
              .fontColor(GREEN)
              .backgroundColor('#E5EEE1')
              .borderRadius(10)
              .height(40)
              .onClick(() => this.simulateMicrophone())
            if (this.simulation.length > 0) {
              Text(this.simulation)
                .fontSize(13)
                .fontColor(this.selected.includes(1) && !this.unknown ? GREEN : '#9E3B31')
            }
          }
          .width('100%')
          .alignItems(HorizontalAlign.Start)
          .padding(18)
          .backgroundColor(WHITE)
          .border({ width: 1, color: LINE, radius: 18 })
        }

        Text('结果仅用于演示。App 不会请求或访问这些资源。')
          .fontSize(12)
          .fontColor(MUTED)
          .padding({ bottom: 12 })
      }
      .width('100%')
      .constraintSize({ maxWidth: 620 })
      .padding({ top: 20, bottom: 16, left: 20, right: 20 })
      .alignItems(HorizontalAlign.Start)
    }
    .align(Alignment.Top)
    .width('100%')
    .height('100%')
    .backgroundColor(CANVAS)
  }
}
